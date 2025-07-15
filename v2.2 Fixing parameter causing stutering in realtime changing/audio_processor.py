import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as tat
import noisereduce as nr
import librosa
import torch.nn.functional as F
import time, traceback, sys, threading, queue

class AudioProcessor:
    def __init__(self, config, rvc, gui_window):
        self.config = config
        self.rvc = rvc
        self.window = gui_window
        self.update_queue = queue.Queue()
        self.flag_vc = False
        self.is_file_playing = False
        self.file_audio_data = None
        self.file_playback_position = 0
        self.playback_thread = None
        self._is_device_changing = False
        self.input_wav, self.output_wav_cache, self.pitch, self.pitchf = (None,) * 4
        self.sola_buffer, self.fade_in_window, self.fade_out_window = (None,) * 3
        self.resampler, self.to_model_resampler, self.from_model_resampler = (None,) * 3
        self.input_tensor, self.resampled_input, self.processed_output_model = (None,) * 3

    def start(self):
        self.flag_vc = True
        self._initialize_buffers()
        if self.config.input_source == 'microphone':
            thread_vc = threading.Thread(target=self._soundinput)
            thread_vc.start()

    def stop(self):
        self.flag_vc = False
        self.is_file_playing = False
        if getattr(self.playback_thread, 'is_alive', False): self.playback_thread.join()
        if self.config.enable_cmd_diagnostics: print("\nDiagnostic monitoring stopped.")
    
    def queue_parameter_update(self, key, value):
        self.update_queue.put((key, value))
        
    def _apply_queued_updates(self):
        while not self.update_queue.empty():
            try:
                key, value = self.update_queue.get_nowait()
                self.rvc.update_parameter(key, value)
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            except queue.Empty:
                break

    def play_file(self):
        if self.is_file_playing: return
        if self.file_audio_data is not None:
            self.is_file_playing = True
            self.playback_thread = threading.Thread(target=self._file_playback_thread)
            self.playback_thread.start()
            self.window.write_event_value('-FILE_PLAYBACK_STARTED-', '')

    def stop_file(self):
        self.is_file_playing = False
        self.file_playback_position = 0
        self.window.write_event_value('-FILE_PLAYBACK_STOPPED-', '')

    def set_loop(self, loop_status: bool):
        self.queue_parameter_update("loop_audio_file", loop_status)

    def request_load_file(self, path):
        self.stop_file()
        self.window.write_event_value('-AUDIO_FILE_LOADING-', '')
        loader_thread = threading.Thread(target=self._load_file_thread_target, args=(path,))
        loader_thread.start()

    def _load_file_thread_target(self, path):
        try:
            self.config.input_audio_path = path
            self.file_audio_data, _ = librosa.load(path, sr=self.config.device_samplerate, mono=True)
            self.file_playback_position = 0
            self.window.write_event_value('-AUDIO_FILE_LOADED-', (True, "Success"))
        except Exception as e:
            self.file_audio_data = None
            self.window.write_event_value('-AUDIO_FILE_LOADED-', (False, str(e)))
        
    def change_output_device(self):
        if self.is_file_playing:
            self._is_device_changing = True
            self.is_file_playing = False
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join()
            self._is_device_changing = False
            self.play_file()

    def _initialize_buffers(self):
        device = self.config.device
        model_sr = self.rvc.tgt_sr
        device_sr = self.config.device_samplerate
        
        if model_sr != device_sr:
            self.to_model_resampler = tat.Resample(orig_freq=device_sr, new_freq=model_sr).to(device)
            self.from_model_resampler = tat.Resample(orig_freq=model_sr, new_freq=device_sr).to(device)
        
        self.block_frame_device = int(self.config.block_time * device_sr)
        self.block_frame_model = int(self.config.block_time * model_sr)
        
        crossfade_time = min(self.config.crossfade_time, self.config.block_time)
        self.crossfade_frame = int(crossfade_time * model_sr)
        self.sola_search_frame = int(self.config.sola_search_ms * model_sr)
        self.extra_frame = int(self.config.extra_time * model_sr)
        self.zc = model_sr // 100
        
        buffer_size_frames = int(np.ceil((self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame_model) / self.zc) * self.zc)
        
        self.input_wav = np.zeros(buffer_size_frames, dtype=np.float32)
        self.output_wav_cache = torch.zeros(buffer_size_frames, device=device, dtype=torch.float32)
        self.pitch = np.zeros(buffer_size_frames // self.zc, dtype="int32")
        self.pitchf = np.zeros(buffer_size_frames // self.zc, dtype="float64")
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=device, dtype=torch.float32)
        self.fade_in_window = torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=device, dtype=torch.float32)
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=model_sr, new_freq=16000, dtype=torch.float32).to(device)
        self.input_tensor = torch.zeros(buffer_size_frames, device=device, dtype=torch.float32)
        self.resampled_input = None 
        self.processed_output_model = torch.zeros(self.block_frame_model, device=device, dtype=torch.float32)

    def _soundinput(self):
        try:
            with sd.Stream(
                channels=1,
                callback=self._mic_callback,
                blocksize=self.block_frame_device,
                samplerate=self.config.device_samplerate,
                dtype="float32"
            ):
                while self.flag_vc: time.sleep(self.config.block_time)
        except Exception as e:
            self.window.write_event_value('-STREAM_ERROR-', str(e))

    def _mic_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        if status.output_underflow: print("Output underflow")
        outdata[:] = self._process_chunk(indata[:, 0]).reshape(-1, 1)

    def _file_playback_thread(self):
        try:
            with sd.OutputStream(channels=1, samplerate=self.config.device_samplerate, blocksize=self.block_frame_device) as stream:
                while self.is_file_playing and self.flag_vc:
                    loop = self.config.loop_audio_file
                    start, end = self.file_playback_position, self.file_playback_position + self.block_frame_device

                    if start >= len(self.file_audio_data):
                        if loop: self.file_playback_position = 0; continue
                        else: break
                    
                    chunk = self.file_audio_data[start:end]
                    if len(chunk) < self.block_frame_device:
                         chunk = np.pad(chunk, (0, self.block_frame_device - len(chunk)))

                    output_chunk = self._process_chunk(chunk).reshape(-1, 1)
                    stream.write(output_chunk)
                    self.file_playback_position = end
        except Exception as e:
            self.window.write_event_value('-STREAM_ERROR-', str(e))
        finally:
            if not self._is_device_changing: self.stop_file()

    def _process_chunk(self, indata_mono: np.ndarray) -> np.ndarray:
        self._apply_queued_updates()
        
        input_volume, output_volume, threshold = self.config.input_volume, self.config.output_volume, self.config.threhold
        i_noise, o_noise = self.config.I_noise_reduce, self.config.O_noise_reduce
        samplerate, device_samplerate = self.config.samplerate, self.config.device_samplerate
        enable_diagnostics = self.config.enable_cmd_diagnostics

        try:
            start_time = time.time()
            indata_mono *= input_volume
            
            if self.to_model_resampler: indata_resampled = self.to_model_resampler(torch.from_numpy(indata_mono).to(self.config.device)).cpu().numpy()
            else: indata_resampled = indata_mono
                
            if i_noise: indata_resampled = nr.reduce_noise(y=indata_resampled, sr=samplerate)
            if np.sqrt(np.mean(np.square(indata_resampled))) < (10**(threshold/20)): indata_resampled[:] = 0
            
            self.input_wav = np.roll(self.input_wav, -self.block_frame_model)
            self.input_wav[-self.block_frame_model:] = indata_resampled
            
            self.input_tensor.copy_(torch.from_numpy(self.input_wav))
            self.resampled_input = self.resampler(self.input_tensor)
            
            rate1 = self.block_frame_model / self.input_tensor.shape[0]
            rate2 = (self.crossfade_frame + self.sola_search_frame + self.block_frame_model) / self.input_tensor.shape[0]
            
            rms = np.sqrt(np.mean(np.square(indata_resampled)))
            res2 = self.rvc.infer(self.resampled_input, self.resampled_input[-self.block_frame_model:].cpu().numpy(), rate1, rate2, self.pitch, self.pitchf, rms)
            
            self.output_wav_cache = torch.cat((self.output_wav_cache[res2.shape[0]:], res2.to(self.config.device)))
            infer_wav = self.output_wav_cache[-self.crossfade_frame - self.sola_search_frame - self.block_frame_model:]
            
            # --- Corrected and Optimized SOLA Implementation ---
            sola_search_area_np = infer_wav[:self.crossfade_frame + self.sola_search_frame].cpu().numpy()
            sola_buffer_np = self.sola_buffer.cpu().numpy()
            correlation = np.correlate(sola_search_area_np, sola_buffer_np, mode='valid')
            sola_offset = np.argmax(correlation)
            
            # Check if slice is valid before copying
            slice_end = sola_offset + self.block_frame_model
            if slice_end > infer_wav.shape[0]:
                sola_offset = infer_wav.shape[0] - self.block_frame_model

            self.processed_output_model.copy_(infer_wav[sola_offset : sola_offset + self.block_frame_model])
            self.processed_output_model[:self.crossfade_frame] *= self.fade_in_window
            self.processed_output_model[:self.crossfade_frame] += self.sola_buffer
            
            if sola_offset < self.sola_search_frame:
                self.sola_buffer.copy_(infer_wav[-self.sola_search_frame - self.crossfade_frame + sola_offset : -self.sola_search_frame + sola_offset] * self.fade_out_window)
            else:
                self.sola_buffer.copy_(infer_wav[-self.crossfade_frame:] * self.fade_out_window)
                
            if self.from_model_resampler: output_wav_device = self.from_model_resampler(self.processed_output_model)
            else: output_wav_device = self.processed_output_model
                
            output_data = output_wav_device.cpu().numpy() * (10 ** (output_volume / 20))
            if o_noise: output_data = nr.reduce_noise(y=output_data, sr=device_samplerate)
            
            total_time_ms = (time.time() - start_time) * 1000
            self.window.write_event_value('-UPDATE_STATUS-', (total_time_ms, self.rvc.last_avg_hz))
            
            if enable_diagnostics:
                print(self.rvc.get_diagnostics_text(total_time_ms).ljust(160), end='\r')

            return output_data
        except:
            print(traceback.format_exc())
            return np.zeros(self.block_frame_device, dtype=np.float32)