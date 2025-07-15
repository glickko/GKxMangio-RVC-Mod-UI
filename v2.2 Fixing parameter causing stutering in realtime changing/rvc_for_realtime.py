import faiss, torch, traceback, parselmouth, numpy as np, threading
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import os, sys
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
try:
    import pedalboard
except ImportError:
    pedalboard = None

now_dir = os.getcwd()
sys.path.append(now_dir)
from config import Config

def get_note_to_hz_map():
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_map = {}
    for octave in range(9):
        for i, note in enumerate(note_names):
            freq = 440.0 * (2.0 ** ((octave - 4) + (i - 9) / 12.0))
            note_map[f"{note}{octave}"] = freq
    return note_map

class RVC:
    def __init__(self, gui_config) -> None:
        self.param_lock = threading.Lock()
        
        for key, value in gui_config.__dict__.items():
            setattr(self, key, value)
        
        self.note_to_hz = get_note_to_hz_map()
        self.last_adjustment = 0.0
        self.last_avg_hz = 0.0
        self.sr = 16000
        self.window = 160
        self.pitch_adj_semitones = 0.0
        self.effects_diag = {} 
        self.last_updated_param_info = None # For diagnostic confirmation
        
        self.split_pitch_crossover_hz = self.note_to_hz.get(self.split_pitch_crossover, 261.63)
        
        self.pre_allocate_effects()
        
        try:
            if self.index_path is not None and os.path.exists(self.index_path) and self.index_rate > 0:
                self.index = faiss.read_index(self.index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            else:
                self.index = None

            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="")
            hubert_model = models[0]
            hubert_model = hubert_model.to(self.device)
            if self.is_half: hubert_model = hubert_model.half()
            else: hubert_model = hubert_model.float()
            hubert_model.eval()
            self.model = hubert_model
            cpt = torch.load(self.pth_path, map_location="cpu")
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            model_class = {
                ("v1", 1): SynthesizerTrnMs256NSFsid, ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                ("v2", 1): SynthesizerTrnMs768NSFsid, ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
            }[(self.version, self.if_f0)]
            self.net_g = model_class(*cpt["config"], is_half=self.is_half)
            del self.net_g.enc_q
            self.net_g.load_state_dict(cpt["weight"], strict=False)
            self.net_g.eval().to(self.device)
            if self.is_half: self.net_g = self.net_g.half()
        except Exception as e:
            raise e
            
    def pre_allocate_effects(self):
        self.reverb_effect = None
        self.saturation_effect = None
        self.cave_reverb_effect = None
        self.cave_delay_effect = None
        self.dyn_prox_bass_shelf = None
        self.dyn_prox_treble_shelf = None
        self.dyn_prox_reverb = None
        if pedalboard is not None:
            self.reverb_effect = pedalboard.Reverb(room_size=0.5, damping=0.5, wet_level=0.33, dry_level=0.4)
            self.saturation_effect = pedalboard.Distortion(drive_db=6.0)
            self.cave_reverb_effect = pedalboard.Reverb(room_size=0.8, damping=0.5, wet_level=0.5, dry_level=0.5)
            self.cave_delay_effect = pedalboard.Delay(delay_seconds=0.25, feedback=0.4, mix=0.5)
            self.dyn_prox_bass_shelf = pedalboard.LowShelfFilter(cutoff_frequency_hz=250, gain_db=0.0)
            self.dyn_prox_treble_shelf = pedalboard.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=0.0)
            self.dyn_prox_reverb = pedalboard.Reverb(room_size=0.2, wet_level=0.0, dry_level=1.0)

    def update_parameter(self, key, value):
        with self.param_lock:
            if hasattr(self, key):
                setattr(self, key, value)
            
            if key == "split_pitch_crossover":
                self.split_pitch_crossover_hz = self.note_to_hz.get(value, 261.63)

            self.last_updated_param_info = (key, ttime())
            self._update_effect_params(key)

    def _update_effect_params(self, key):
        if pedalboard is None: return
        
        if "reverb" in key and self.reverb_effect:
            self.reverb_effect.room_size = self.reverb_room_size
            self.reverb_effect.damping = self.reverb_damping
            self.reverb_effect.wet_level = self.reverb_wet_level
            self.reverb_effect.dry_level = self.reverb_dry_level
        if "saturation" in key and self.saturation_effect:
            self.saturation_effect.drive_db = self.saturation_drive_db
        if "cave" in key:
            if self.cave_reverb_effect:
                self.cave_reverb_effect.wet_level = self.cave_mix
                self.cave_reverb_effect.dry_level = 1.0 - self.cave_mix
            if self.cave_delay_effect:
                self.cave_delay_effect.delay_seconds = self.cave_delay_time / 1000.0
                self.cave_delay_effect.feedback = self.cave_feedback
                self.cave_delay_effect.mix = self.cave_mix
        if "dynamic_proximity_room_size" in key and self.dyn_prox_reverb:
            self.dyn_prox_reverb.room_size = self.dynamic_proximity_room_size

    def _apply_auto_pitch_correction(self, f0, base_f0_up_key, auto_pitch_settings):
        use_auto_pitch, stability, strength, max_adj, use_split, split_hz, low_strength, low_max, high_strength, high_max, use_shout, shout_strength, voice_profile = auto_pitch_settings
        if not use_auto_pitch:
            self.pitch_adj_semitones = 0.0
            return base_f0_up_key

        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) < 5:
            self.pitch_adj_semitones = self.last_adjustment
            return base_f0_up_key + self.last_adjustment

        average_f0_hz, self.last_avg_hz = np.mean(voiced_f0), np.mean(voiced_f0)
        
        current_strength, current_max_adjustment = (strength, max_adj)
        self.effects_diag['pitch_range'] = 'Global'
        if use_split and average_f0_hz < split_hz:
            current_strength, current_max_adjustment = (low_strength, low_max)
            self.effects_diag['pitch_range'] = 'Low'
        elif use_split:
            current_strength, current_max_adjustment = (high_strength, high_max)
            self.effects_diag['pitch_range'] = 'High'
        
        self.effects_diag['ap_strength'] = current_strength
        self.effects_diag['ap_max_adj'] = current_max_adjustment

        adjustment = 0.0
        if use_shout and voice_profile in ["Male to Female", "Female to Male"] and average_f0_hz > 250.0:
            adjustment = - (12 * np.log2(average_f0_hz / 250.0)) * shout_strength
        else:
            neutral_pitch_hz = 190.0 if voice_profile == "Male to Female" else 120.0 if voice_profile == "Female to Male" else 0.0
            if neutral_pitch_hz > 0 and average_f0_hz > 1.0:
                adjustment = - (12 * np.log2(average_f0_hz / neutral_pitch_hz)) * current_strength
            else:
                self.pitch_adj_semitones = self.last_adjustment
                return base_f0_up_key + self.last_adjustment

        adjustment = np.clip(adjustment, self.last_adjustment - 0.5, self.last_adjustment + 0.5)
        adjustment = np.clip(adjustment, -current_max_adjustment, current_max_adjustment)
        
        smoothed_adjustment = self.last_adjustment * stability + adjustment * (1 - stability)
        self.last_adjustment, self.pitch_adj_semitones = smoothed_adjustment, smoothed_adjustment
        return base_f0_up_key + smoothed_adjustment

    def _apply_effects(self, audio_np: np.ndarray, rms_level: float, effects_settings):
        (
            enable_low_damp, low_damp_hz, low_damp_db, enable_dyn_prox, dyn_prox_strength,
            enable_discord, discord_qual, discord_prox, discord_noise, enable_phone,
            enable_sat, sat_hz, enable_cave, use_reverb
        ) = effects_settings
        
        pitch_range = self.effects_diag.get('pitch_range', 'Global')
        ap_strength = self.effects_diag.get('ap_strength', 0.0)
        ap_max_adj = self.effects_diag.get('ap_max_adj', 0.0)
        self.effects_diag = {'pitch_range': pitch_range, 'ap_strength': ap_strength, 'ap_max_adj': ap_max_adj}

        if enable_low_damp:
            self.effects_diag['low_damp'] = "Off"
            if 0 < self.last_avg_hz < low_damp_hz:
                audio_np *= (10 ** (low_damp_db / 20)); self.effects_diag['low_damp'] = f"On ({low_damp_db:.1f}dB)"
        
        if enable_dyn_prox and self.dyn_prox_bass_shelf:
            prox_factor = np.clip(np.interp(20*np.log10(max(rms_level, 1e-6)), [-50, -10], [0.0, 1.0]), 0.0, 1.0)
            self.effects_diag['dyn_prox'] = prox_factor
            audio_np *= (1.0 - (1.0 - prox_factor) * dyn_prox_strength * 0.5)
            self.dyn_prox_bass_shelf.gain_db = prox_factor * dyn_prox_strength * 6.0
            self.dyn_prox_treble_shelf.gain_db = (1.0 - prox_factor) * dyn_prox_strength * -9.0
            self.dyn_prox_reverb.wet_level = (1.0 - prox_factor) * dyn_prox_strength * 0.5
            self.dyn_prox_reverb.dry_level = 1.0 - self.dyn_prox_reverb.wet_level
            audio_np = self.dyn_prox_reverb(self.dyn_prox_treble_shelf(self.dyn_prox_bass_shelf(audio_np, self.tgt_sr), self.tgt_sr), self.tgt_sr)

        if enable_discord:
            volume_scale = 0.4 + discord_prox * 0.6; audio_np *= volume_scale
            self.effects_diag.update({"prox_vol": volume_scale, "prox_hp": "N/A", "quality_lp": "N/A"})
            if discord_qual < 1.0:
                cutoff = 3000 + discord_qual * (min(18000, self.tgt_sr / 2.1) - 3000)
                self.effects_diag["quality_lp"] = f"{cutoff:.0f}Hz"
                if cutoff < self.tgt_sr/2: audio_np = signal.lfilter(*signal.butter(4, cutoff, 'low', fs=self.tgt_sr), audio_np)
            if discord_prox < 0.5:
                cutoff = 400 - (discord_prox / 0.5) * 300
                self.effects_diag["prox_hp"] = f"{cutoff:.0f}Hz"
                if cutoff > 0: audio_np = signal.lfilter(*signal.butter(2, cutoff, 'high', fs=self.tgt_sr), audio_np)
            if discord_noise > -80.0: audio_np += np.random.randn(len(audio_np)) * (10 ** (discord_noise / 20))
        
        if enable_phone:
            audio_np = signal.lfilter(*signal.butter(4, [300, 3400], 'bandpass', fs=self.tgt_sr), audio_np)
            audio_np = np.tanh(audio_np * 1.2) / 1.2
        
        if enable_sat and self.saturation_effect:
            self.effects_diag['saturation'] = "Off"
            if self.last_avg_hz > sat_hz:
                audio_np = self.saturation_effect(audio_np, self.tgt_sr)
                self.effects_diag['saturation'] = f"On ({self.saturation_effect.drive_db:.1f}dB)"
        
        if enable_cave and self.cave_reverb_effect:
            audio_np = self.cave_reverb_effect(audio_np, self.tgt_sr)
            audio_np = self.cave_delay_effect(audio_np, self.tgt_sr)

        if use_reverb and self.reverb_effect: audio_np = self.reverb_effect(audio_np, self.tgt_sr)

        return np.clip(audio_np, -1.0, 1.0)
    
    def get_f0(self, x, f0_up_key, rmvpe_threshold, auto_pitch_settings):
        if not hasattr(self, "model_rmvpe"):
            from rmvpe import RMVPE
            self.model_rmvpe = RMVPE("rmvpe.pt", is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio(x, thred=rmvpe_threshold)
        
        if np.any(f0 > 0): self.last_avg_hz = np.mean(f0[f0 > 0])
        else: self.last_avg_hz = 0.0

        final_f0_up_key = self._apply_auto_pitch_correction(f0, f0_up_key, auto_pitch_settings)
        f0 *= pow(2, final_f0_up_key / 12)

        f0_min, f0_max = 50, 1100
        f0_mel_min, f0_mel_max = 1127 * np.log(1 + f0_min / 700), 1127 * np.log(1 + f0_max / 700)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1; f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int32), f0.copy()

    def get_diagnostics_text(self, latency_ms):
        diag_text = ""
        if self.last_updated_param_info:
            param_key, update_time = self.last_updated_param_info
            if ttime() - update_time < 1.5:
                diag_text += f"Applied: {param_key} | "
            else:
                self.last_updated_param_info = None

        with self.param_lock:
            diag_text += f"Status: Running | Latency: {latency_ms:.1f}ms | Avg F0: {self.last_avg_hz:.1f}Hz | Pitch Adj: {self.pitch_adj_semitones:.2f}st | "
            if self.auto_pitch_correction:
                strength = self.effects_diag.get('ap_strength', self.auto_pitch_strength)
                max_adj = self.effects_diag.get('ap_max_adj', self.auto_pitch_max_adjustment)
                if self.use_split_pitch_correction:
                    active_range = self.effects_diag.get('pitch_range', 'N/A')
                    diag_text += f"APitch({active_range}): Str:{strength:.2f} Max:{max_adj:.1f}st | "
                else:
                    diag_text += f"APitch(Global): Str:{strength:.2f} Max:{max_adj:.1f}st | "
            if self.enable_dynamic_proximity: diag_text += f"DynProx: {self.effects_diag.get('dyn_prox', 0):.2f} | "
            if self.enable_low_freq_dampening: diag_text += f"Low Freq Damp: {self.effects_diag.get('low_damp', 'Off')} | "
            if self.enable_saturation_effect: diag_text += f"Saturation: {self.effects_diag.get('saturation', 'Off')} | "
        return diag_text

    def infer(self, feats: torch.Tensor, indata: np.ndarray, rate1, rate2, cache_pitch, cache_pitchf, rms_level: float) -> np.ndarray:
        with self.param_lock:
            pitch_val, index_rate_val, formant_val, timbre_val, voice_profile_val = self.pitch, self.index_rate, self.formant_shift, self.timbre, self.voice_profile
            auto_pitch_settings = (self.auto_pitch_correction, self.pitch_stability, self.auto_pitch_strength, self.auto_pitch_max_adjustment, self.use_split_pitch_correction, self.split_pitch_crossover_hz, self.low_pitch_strength, self.low_pitch_max_adjustment, self.high_pitch_strength, self.high_pitch_max_adjustment, self.use_shout_dampening, self.shout_dampening_strength, voice_profile_val)
            effects_settings = (self.enable_low_freq_dampening, self.low_freq_dampening_threshold_hz, self.low_freq_dampening_level_db, self.enable_dynamic_proximity, self.dynamic_proximity_strength, self.enable_discord_effects, self.discord_quality, self.discord_proximity, self.discord_noise, self.enable_phone_effect, self.enable_saturation_effect, self.saturation_threshold_hz, self.enable_cave_effect, self.use_reverb)

        feats = feats.view(1, -1)
        if self.is_half: feats = feats.half()
        else: feats = feats.float()
        feats = feats.to(self.device)
        
        with torch.no_grad():
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 9 if self.version == "v1" else 12}
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
        
        if self.index is not None and index_rate_val > 0:
            leng_replace_head = int(rate1 * feats[0].shape[0])
            npy = feats[0][-leng_replace_head:].cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            weight = np.square(1 / score); weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half: npy = npy.astype("float16")
            feats[0][-leng_replace_head:] = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate_val + (1 - index_rate_val) * feats[0][-leng_replace_head:])

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        
        p_len = min(feats.shape[1], 13000)
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(indata, pitch_val, 0.03, auto_pitch_settings)
            if voice_profile_val != "Default":
                pitchf *= formant_val
                pitchf += (timbre_val - 1.0) * self.sr / self.window
            p_len = min(p_len, cache_pitch.shape[0])
            cache_pitch[-pitch.shape[0]:] = pitch; cache_pitchf[-pitchf.shape[0]:] = pitchf
        
        feats = feats[:, :p_len, :]
        p_len_tensor, sid = torch.LongTensor([p_len]).to(self.device), torch.LongTensor([0]).to(self.device)
        
        with torch.no_grad():
            if self.if_f0 == 1:
                local_cache_pitch = torch.LongTensor(cache_pitch[:p_len]).unsqueeze(0).to(self.device)
                local_cache_pitchf = torch.FloatTensor(cache_pitchf[:p_len]).unsqueeze(0).to(self.device)
                infered_audio = self.net_g.infer(feats, p_len_tensor, local_cache_pitch, local_cache_pitchf, sid, rate2)[0][0, 0]
            else:
                infered_audio = self.net_g.infer(feats, p_len_tensor, sid, rate2)[0][0, 0]
        
        audio_np = self._apply_effects(infered_audio.data.cpu().float().numpy(), rms_level, effects_settings)
        return torch.from_numpy(audio_np.astype(np.float32))