import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as tat
import noisereduce as nr
import librosa
import torch.nn.functional as F
import time, traceback, sys

class AudioProcessor:
    def __init__(self, config, rvc, gui_window):
        self.config = config
        self.rvc = rvc
        self.window = gui_window
        self.flag_vc = False
        
        self.input_wav = None
        self.output_wav_cache = None
        self.pitch = None
        self.pitchf = None
        self.sola_buffer = None
        self.fade_in_window = None
        self.fade_out_window = None
        self.resampler = None
        self.to_model_resampler = None
        self.from_model_resampler = None

    def start(self):
        self.flag_vc = True
        self._initialize_buffers()
        
        import threading
        thread_vc = threading.Thread(target=self._soundinput)
        thread_vc.start()

    def stop(self):
        self.flag_vc = False
        if self.config.enable_cmd_diagnostics:
            print("\nDiagnostic monitoring stopped.")

    def _initialize_buffers(self):
        device = self.config.device
        model_sr = self.rvc.tgt_sr
        device_sr = self.config.device_samplerate
        
        print(f"Model SR: {model_sr} Hz | Device SR: {device_sr} Hz")
        
        if model_sr != device_sr:
            print("Resampling required. Creating resamplers.")
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

    def _soundinput(self):
        channels = 1 if sys.platform == "darwin" else 2
        try:
            with sd.Stream(
                channels=channels,
                callback=self.audio_callback,
                blocksize=self.block_frame_device,
                samplerate=self.config.device_samplerate,
                dtype="float32"
            ):
                while self.flag_vc:
                    time.sleep(self.config.block_time)
            print("Audio stream stopped.")
        except Exception as e:
            print("\nError in audio stream, please check your audio devices.")
            print(traceback.format_exc())
            self.window.write_event_value('-STREAM_ERROR-', '')


    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        try:
            start_time = time.perf_counter()
            device = self.config.device
            indata_mono = librosa.to_mono(indata.T) * self.config.input_volume
            
            if self.to_model_resampler is not None:
                indata_resampled = self.to_model_resampler(torch.from_numpy(indata_mono).to(device)).cpu().numpy()
            else:
                indata_resampled = indata_mono
                
            if self.config.I_noise_reduce:
                indata_resampled = nr.reduce_noise(y=indata_resampled, sr=self.config.samplerate)
                
            rms = np.sqrt(np.mean(np.square(indata_resampled)))
            db_threshold_val = 20 * np.log10(rms) < self.config.threhold
            if self.config.threhold > -60 and db_threshold_val:
                indata_resampled[:] = 0
            
            self.input_wav = np.roll(self.input_wav, -self.block_frame_model)
            self.input_wav[-self.block_frame_model:] = indata_resampled
            
            inp = torch.from_numpy(self.input_wav).to(device)
            res1 = self.resampler(inp)
            rate1 = self.block_frame_model / inp.shape[0]
            rate2 = (self.crossfade_frame + self.sola_search_frame + self.block_frame_model) / inp.shape[0]
            
            # Pass the calculated RMS level to the infer method
            res2 = self.rvc.infer(res1, res1[-self.block_frame_model:].cpu().numpy(), rate1, rate2, self.pitch, self.pitchf, self.config.f0method, rms)
            
            self.output_wav_cache = torch.cat((self.output_wav_cache[res2.shape[0]:], res2.to(device)))
            infer_wav = self.output_wav_cache[-self.crossfade_frame - self.sola_search_frame - self.block_frame_model:]
            
            cor_nom = F.conv1d(infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame], self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame] ** 2, torch.ones(1, 1, self.crossfade_frame, device=device)) + 1e-8)
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            
            output_wav_model = infer_wav[sola_offset : sola_offset + self.block_frame_model]
            output_wav_model[:self.crossfade_frame] *= self.fade_in_window
            output_wav_model[:self.crossfade_frame] += self.sola_buffer
            
            if sola_offset < self.sola_search_frame:
                self.sola_buffer[:] = infer_wav[-self.sola_search_frame - self.crossfade_frame + sola_offset : -self.sola_search_frame + sola_offset] * self.fade_out_window
            else:
                self.sola_buffer[:] = infer_wav[-self.crossfade_frame:] * self.fade_out_window
                
            if self.from_model_resampler is not None:
                output_wav_device = self.from_model_resampler(output_wav_model)
            else:
                output_wav_device = output_wav_model
                
            output_data = output_wav_device.cpu().numpy() * (10 ** (self.config.output_volume / 20))
            
            if self.config.O_noise_reduce:
                output_data = nr.reduce_noise(y=output_data, sr=self.config.device_samplerate)
                
            outdata[:] = np.tile(output_data[:len(outdata)], (outdata.shape[1], 1)).T if outdata.ndim > 1 else output_data[:len(outdata)]
            
            total_time_ms = (time.perf_counter() - start_time) * 1000
            avg_hz = self.rvc.last_avg_hz
            self.window.write_event_value('-UPDATE_STATUS-', (total_time_ms, avg_hz))

            if self.config.enable_cmd_diagnostics:
                diag_text = "Status: Running | "
                diag_text += f"Latency: {total_time_ms:.1f}ms | "
                diag_text += f"Avg F0: {avg_hz:.1f}Hz | "
                diag_text += f"Pitch Adj: {self.rvc.pitch_adj_semitones:.2f}st | "
                if self.config.enable_dynamic_proximity:
                    effects = self.rvc.effects_diag
                    diag_text += f"DynProx: {effects.get('dyn_prox', 0):.2f} | "
                if self.config.enable_low_freq_dampening:
                    effects = self.rvc.effects_diag
                    diag_text += f"Low Freq Damp: {effects.get('low_damp', 'Off')} | "
                if self.config.enable_saturation_effect:
                    effects = self.rvc.effects_diag
                    diag_text += f"Saturation: {effects.get('saturation', 'Off')} | "
                if self.config.enable_discord_effects:
                    effects = self.rvc.effects_diag
                    diag_text += f"Prox Vol: {effects.get('prox_vol', 0):.2f} | "
                    diag_text += f"Prox HPF: {effects.get('prox_hp', 'N/A')} | "
                    diag_text += f"Quality LPF: {effects.get('quality_lp', 'N/A')} | "
                    diag_text += f"Noise: {self.config.discord_noise:.1f}dB | "
                if self.config.enable_phone_effect:
                    diag_text += "Phone: On | "
                if self.config.enable_cave_effect:
                    diag_text += f"Cave Mix: {self.config.cave_mix * 100:.0f}% | "
                
                # Use carriage return to print on the same line
                print(diag_text.ljust(150), end='\r')


        except Exception as e:
            print(traceback.format_exc())