# audio_processor.py (FINAL UNIFIED SHUTDOWN)
import os
import numpy as np
import traceback
import sounddevice as sd
import torch
import torchaudio.transforms as tat
import noisereduce as nr
import torch.nn.functional as F
import time, traceback, sys, threading, queue, math, subprocess
try:
    import pedalboard
except ImportError:
    pedalboard = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

class AudioProcessor:
    def __init__(self, config, rvc, gui_window):
        self.config = config
        self.rvc = rvc
        self.window = gui_window
        self.update_queue = queue.Queue()
        self.file_audio_queue = queue.Queue(maxsize=10)
        self.flag_vc = False
        self.playback_mode = None
        self.file_audio_path = None
        self.playback_thread = None
        self.ffmpeg_process = None
        self.direct_stream = None
        self.main_stream = None
        self.lock = threading.Lock()
        self.input_wav, self.output_wav_cache, self.pitch, self.pitchf = (None,) * 4
        self.sola_buffer, self.fade_in_window, self.fade_out_window = (None,) * 3
        self.resampler, self.to_model_resampler, self.from_model_resampler = (None,) * 3
        self.input_tensor, self.resampled_input, self.processed_output_model = (None,) * 3
        self.input_pitch_shifter = None
        self.input_timbre_shifter = None
        if pedalboard is not None:
            self.input_pitch_shifter = pedalboard.PitchShift()
            self.input_timbre_shifter = pedalboard.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=0.0, q=0.707)

    def start(self):
        self.flag_vc = True
        self._initialize_buffers()
        try:
            sd.check_input_settings(device=sd.default.device[0], samplerate=self.config.device_samplerate)
            sd.check_output_settings(device=sd.default.device[1], samplerate=self.config.device_samplerate)
            
            self.main_stream = sd.Stream(
                device=(sd.default.device[0], sd.default.device[1]),
                samplerate=self.config.device_samplerate,
                blocksize=self.block_frame_device,
                channels=(1, 1),
                dtype='float32',
                callback=self._audio_callback
            )
            self.main_stream.start()
            print("Audio streams started successfully using callback architecture.")
        except Exception as e:
            print(f"!!! FAILED TO START AUDIO STREAM !!!\n{'='*60}\n{traceback.format_exc()}\n{'='*60}")
            self.window.write_event_value('-STREAM_ERROR-', str(e))

    def stop(self):
        self.flag_vc = False
        self.stop_file()
        
        if self.main_stream and self.main_stream.active:
            self.main_stream.stop()
            self.main_stream.close()
            self.main_stream = None
            print("Main audio streams stopped and closed.")
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
            
        if self.config.enable_cmd_diagnostics: print("\nDiagnostic monitoring stopped.")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        if not self.flag_vc:
            outdata[:] = 0
            return
        if status:
            print(status, file=sys.stderr)
        
        try:
            combined_chunk = indata[:, 0].copy()

            try:
                file_chunk = self.file_audio_queue.get_nowait()
                volume_multiplier = 10 ** (self.config.file_input_volume_db / 20)
                processed_file_chunk = file_chunk * volume_multiplier

                if len(processed_file_chunk) < len(combined_chunk):
                   processed_file_chunk = np.pad(processed_file_chunk, (0, len(combined_chunk) - len(processed_file_chunk)))
                combined_chunk += processed_file_chunk[:len(combined_chunk)]
            except queue.Empty:
                pass

            np.clip(combined_chunk, -1.0, 1.0, out=combined_chunk)
            
            processed_chunk = self._process_chunk(combined_chunk)
            outdata[:] = processed_chunk.reshape(-1, 1)
        except Exception:
            print(traceback.format_exc())
            outdata[:] = 0

    def is_any_file_playing(self):
        return self.playback_mode is not None

    def queue_parameter_update(self, key, value):
        self.update_queue.put((key, value))
        
    def _apply_queued_updates(self):
        while not self.update_queue.empty():
            try:
                key, value = self.update_queue.get_nowait()
                if hasattr(self.rvc, key): self.rvc.update_parameter(key, value)
                if hasattr(self.config, key): setattr(self.config, key, value)
            except queue.Empty: break
    
    def request_load_file(self, path):
        with self.lock:
            self.stop_file(acquire_lock=False)
            self.file_audio_path = path
            print(f"File loaded and ready for playback: {os.path.basename(path)}")
    
    def play_file_inference(self):
        with self.lock:
            if self.is_any_file_playing() or not self.file_audio_path or not self.flag_vc: return
            self.playback_mode = 'inference'
            self.playback_thread = threading.Thread(target=self._ffmpeg_playback_thread, daemon=True)
            self.playback_thread.start()

    def play_file_direct(self):
        with self.lock:
            if self.is_any_file_playing() or not self.file_audio_path or not self.flag_vc: return
            self.playback_mode = 'direct'
            self.playback_thread = threading.Thread(target=self._direct_playback_thread, daemon=True)
            self.playback_thread.start()

    def stop_file(self, acquire_lock=True):
        if acquire_lock: self.lock.acquire()
        try:
            if not self.is_any_file_playing(): return
            
            current_mode = self.playback_mode
            self.playback_mode = None

            if self.ffmpeg_process:
                try: self.ffmpeg_process.kill()
                except Exception: pass
                self.ffmpeg_process = None

            if current_mode == 'inference':
                while not self.file_audio_queue.empty():
                    try: self.file_audio_queue.get_nowait()
                    except queue.Empty: break
                print("Inference file playback stopped.")
            
            if current_mode == 'direct' and self.direct_stream:
                self.direct_stream.abort()
                self.direct_stream.close()
                self.direct_stream = None
                print("Direct file playback stopped.")
        finally:
            if acquire_lock: self.lock.release()

    def _ffmpeg_playback_thread(self):
        print("Starting FFmpeg playback thread (Inference Mode)...")
        while self.playback_mode == 'inference':
            try:
                process_args = ['ffmpeg', '-v', 'fatal', '-i', self.file_audio_path, '-f', 'f32le', '-ac', '1', '-ar', str(self.config.device_samplerate),'-']
                self.ffmpeg_process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                while self.playback_mode == 'inference':
                    chunk = self.ffmpeg_process.stdout.read(self.block_frame_device * 4)
                    if not chunk: break
                    audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                    self.file_audio_queue.put(audio_chunk, timeout=2)

                if self.ffmpeg_process: self.ffmpeg_process.wait()
            except queue.Full: print("Inference audio queue is full, playback may be lagging."); time.sleep(0.1); continue
            except Exception as e: print(f"Inference playback error: {e}"); break
            
            with self.lock:
                if not self.config.loop_audio_file or self.playback_mode != 'inference': break
                else: print("Looping file (Inference)..."); time.sleep(0.1)
        
        # --- NEXUS FIX: Remove self-calling stop_file() from this thread too ---
        # self.stop_file() 
        self.playback_mode = None
        print("FFmpeg playback thread (Inference) finished.")
    
    def _direct_playback_thread(self):
        print("Starting FFmpeg playback thread (Direct Mode)...")
        try:
            self.direct_stream = sd.OutputStream(samplerate=self.config.device_samplerate, blocksize=self.block_frame_device, channels=1, dtype='float32')
            self.direct_stream.start()
        except Exception as e:
            print(f"Failed to open direct audio stream: {e}"); self.stop_file(); return

        while self.playback_mode == 'direct':
            try:
                process_args = ['ffmpeg', '-v', 'fatal', '-i', self.file_audio_path, '-f', 'f32le', '-ac', '1', '-ar', str(self.config.device_samplerate), '-']
                self.ffmpeg_process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                while self.playback_mode == 'direct':
                    chunk = self.ffmpeg_process.stdout.read(self.block_frame_device * 4)
                    if not chunk: break
                    audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                    
                    if len(audio_chunk) < self.block_frame_device:
                        padded_chunk = np.zeros(self.block_frame_device, dtype=np.float32)
                        padded_chunk[:len(audio_chunk)] = audio_chunk
                        self.direct_stream.write(padded_chunk)
                    else:
                        self.direct_stream.write(audio_chunk)
                
                if self.ffmpeg_process: self.ffmpeg_process.wait()
            except Exception as e: print(f"Direct playback error: {e}"); break
            
            with self.lock:
                if not self.config.loop_audio_file or self.playback_mode != 'direct': break
                else: print("Looping file (Direct)..."); time.sleep(0.1)

        self.playback_mode = None
        print("FFmpeg playback thread (Direct) finished.")

    def _initialize_buffers(self):
        device = self.config.device
        model_sr = self.rvc.tgt_sr
        device_sr = self.config.device_samplerate
        
        if model_sr != device_sr:
            self.to_model_resampler = tat.Resample(orig_freq=device_sr, new_freq=model_sr).to(device)
            self.from_model_resampler = tat.Resample(orig_freq=model_sr, new_freq=device_sr).to(device)
        
        self.block_frame_device = int(self.config.block_time * device_sr)
        self.block_frame_model = int(self.config.block_time * model_sr) if not self.config.bypass_vc else self.block_frame_device
        
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

    def _process_chunk(self, indata_mono: np.ndarray) -> np.ndarray:
        indata_mono = indata_mono.copy()
        
        self._apply_queued_updates()
        input_volume, output_volume = self.config.input_volume, self.config.output_volume
        input_formant, input_timbre = self.config.input_formant_shift, self.config.input_timbre
        threshold, i_noise, o_noise = self.config.threhold, self.config.I_noise_reduce, self.config.O_noise_reduce
        samplerate, device_samplerate = self.config.samplerate, self.config.device_samplerate
        enable_diagnostics = self.config.enable_cmd_diagnostics
        
        try:
            start_time = time.time()
            indata_mono *= input_volume

            if pedalboard is not None and self.input_pitch_shifter is not None and input_formant != 1.0:
                semitones = 12 * math.log2(input_formant) if input_formant > 0 else -12
                self.input_pitch_shifter.semitones = semitones
                indata_mono = self.input_pitch_shifter(indata_mono, device_samplerate)
            
            if pedalboard is not None and self.input_timbre_shifter is not None and input_timbre != 1.0:
                gain_db = (input_timbre - 1.0) * 12
                self.input_timbre_shifter.gain_db = gain_db
                indata_mono = self.input_timbre_shifter(indata_mono, device_samplerate)

            if self.config.enable_input_equalizer and self.rvc.input_equalizer:
                indata_mono = self.rvc.input_equalizer(indata_mono, device_samplerate)

            if self.config.bypass_vc:
                output_data = indata_mono * (10 ** (output_volume / 20))
                if o_noise: output_data = nr.reduce_noise(y=output_data, sr=device_samplerate)
                self.window.write_event_value('-UPDATE_STATUS-', ((time.time() - start_time) * 1000, 0.0))
                return output_data
            
            if self.to_model_resampler:
                indata_resampled = self.to_model_resampler(torch.from_numpy(indata_mono).to(self.config.device)).cpu().numpy()
            else:
                indata_resampled = indata_mono
                
            if i_noise: indata_resampled = nr.reduce_noise(y=indata_resampled, sr=samplerate)
            if np.sqrt(np.mean(np.square(indata_resampled))) < (10**(threshold/20)):
                indata_resampled[:] = 0
            
            self.input_wav = np.roll(self.input_wav, -self.block_frame_model)
            target_len = self.input_wav[-self.block_frame_model:].shape[0]
            self.input_wav[-self.block_frame_model:] = indata_resampled[:target_len]
            
            self.input_tensor.copy_(torch.from_numpy(self.input_wav))
            self.resampled_input = self.resampler(self.input_tensor)
            
            rate1 = self.block_frame_model / self.input_tensor.shape[0]
            rate2 = (self.crossfade_frame + self.sola_search_frame + self.block_frame_model) / self.input_tensor.shape[0]
            
            rms = np.sqrt(np.mean(np.square(indata_resampled)))
            res2 = self.rvc.infer(self.resampled_input, self.resampled_input[-self.block_frame_model:].cpu().numpy(), rate1, rate2, self.pitch, self.pitchf, rms, self.crossfade_frame)
            
            self.output_wav_cache = torch.cat((self.output_wav_cache[res2.shape[0]:], res2.to(self.config.device)))
            infer_wav = self.output_wav_cache[-self.crossfade_frame - self.sola_search_frame - self.block_frame_model:]
            
            sola_search_area_np = infer_wav[:self.crossfade_frame + self.sola_search_frame].cpu().numpy()
            sola_buffer_np = self.sola_buffer.cpu().numpy()
            correlation = np.correlate(sola_search_area_np, sola_buffer_np, mode='valid')
            sola_offset = np.argmax(correlation)
            
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
                
            if self.from_model_resampler:
                output_wav_device = self.from_model_resampler(self.processed_output_model)
            else:
                output_wav_device = self.processed_output_model
                
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