import faiss, torch, traceback, parselmouth, numpy as np
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
    print("Pedalboard library not found. Please install it using: pip install pedalboard")
    pedalboard = None

now_dir = os.getcwd()
sys.path.append(now_dir)
from config import Config
from multiprocessing import Manager as M

mm = M()
config = Config()


def get_note_to_hz_map():
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_map = {}
    for octave in range(9):
        for i, note in enumerate(note_names):
            freq = 440.0 * (2.0 ** ((octave - 4) + (i - 9) / 12.0))
            note_map[f"{note}{octave}"] = freq
    return note_map

class RVC:
    def __init__(
        self,
        key,
        pth_path,
        index_path,
        index_rate,
        device,
        rmvpe_threshold=0.03,
        auto_pitch_correction=False,
        pitch_stability=0.2, 
        auto_pitch_strength=0.3,
        auto_pitch_max_adjustment=2.0,
        voice_profile="Default",
        use_shout_dampening=False,
        shout_dampening_strength=0.8,
        formant_shift=1.0,
        timbre=1.0,
        use_split_pitch_correction=False,
        split_pitch_crossover="C4",
        low_pitch_strength=0.3,
        low_pitch_max_adjustment=2.0,
        high_pitch_strength=0.3,
        high_pitch_max_adjustment=2.0,
        use_reverb=False,
        reverb_room_size=0.5,
        reverb_damping=0.5,
        reverb_wet_level=0.33,
        reverb_dry_level=0.4,
        # Discord Effects
        enable_discord_effects=False,
        discord_proximity=1.0,
        discord_noise=-80.0,
        discord_quality=1.0,
        # Phone and Saturation Effects
        enable_phone_effect=False,
        enable_saturation_effect=False,
        saturation_threshold_hz=800.0,
        saturation_drive_db=6.0,
        # Cave Effect
        enable_cave_effect=False,
        cave_delay_time=250.0,
        cave_feedback=0.4,
        cave_mix=0.5,
        # Low Frequency Dampening
        enable_low_freq_dampening=False,
        low_freq_dampening_threshold_hz=100.0,
        low_freq_dampening_level_db=-6.0,
        # Dynamic Proximity
        enable_dynamic_proximity=False,
        dynamic_proximity_strength=0.5,
        dynamic_proximity_room_size=0.2,
        # Diagnostics
        enable_cmd_diagnostics=False,
    ) -> None:
        try:
            global config
            self.device = device
            self.f0_up_key = key
            self.rmvpe_threshold = rmvpe_threshold
            self.auto_pitch_correction = auto_pitch_correction
            self.pitch_stability = pitch_stability
            self.auto_pitch_strength = auto_pitch_strength
            self.auto_pitch_max_adjustment = auto_pitch_max_adjustment
            self.voice_profile = voice_profile
            self.use_shout_dampening = use_shout_dampening
            self.shout_dampening_strength = shout_dampening_strength
            self.formant_shift = formant_shift
            self.timbre = timbre
            
            self.use_split_pitch_correction = use_split_pitch_correction
            self.note_to_hz = get_note_to_hz_map()
            self.split_pitch_crossover_hz = self.note_to_hz.get(split_pitch_crossover, 261.63)
            self.low_pitch_strength = low_pitch_strength
            self.low_pitch_max_adjustment = low_pitch_max_adjustment
            self.high_pitch_strength = high_pitch_strength
            self.high_pitch_max_adjustment = high_pitch_max_adjustment

            self.last_adjustment = 0.0
            self.last_avg_hz = 0.0
            self.sr = 16000
            self.window = 160
            
            self.use_reverb = use_reverb
            self.reverb_effect = None
            if self.use_reverb and pedalboard is not None:
                self.reverb_effect = pedalboard.Reverb(
                    room_size=reverb_room_size,
                    damping=reverb_damping,
                    wet_level=reverb_wet_level,
                    dry_level=reverb_dry_level
                )
            
            # Discord Effects
            self.enable_discord_effects = enable_discord_effects
            self.discord_proximity = discord_proximity
            self.discord_noise = discord_noise
            self.discord_quality = discord_quality

            # Phone and Saturation Effects
            self.enable_phone_effect = enable_phone_effect
            self.enable_saturation_effect = enable_saturation_effect
            self.saturation_threshold_hz = saturation_threshold_hz
            self.saturation_drive_db = saturation_drive_db
            
            # Cave Effect
            self.enable_cave_effect = enable_cave_effect
            self.cave_delay_time = cave_delay_time
            self.cave_feedback = cave_feedback
            self.cave_mix = cave_mix
            
            # Low Frequency Dampening
            self.enable_low_freq_dampening = enable_low_freq_dampening
            self.low_freq_dampening_threshold_hz = low_freq_dampening_threshold_hz
            self.low_freq_dampening_level_db = low_freq_dampening_level_db

            # Dynamic Proximity
            self.enable_dynamic_proximity = enable_dynamic_proximity
            self.dynamic_proximity_strength = dynamic_proximity_strength
            self.dynamic_proximity_room_size = dynamic_proximity_room_size

            # Diagnostics
            self.enable_cmd_diagnostics = enable_cmd_diagnostics
            self.pitch_adj_semitones = 0.0
            self.effects_diag = {}

            if index_path is not None and os.path.exists(index_path) and index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                print("Index search enabled.")
            else:
                self.index = None
                print("Index search disabled.")

            self.index_rate = index_rate
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                ["hubert_base.pt"],
                suffix="",
            )
            hubert_model = models[0]
            hubert_model = hubert_model.to(config.device)
            if config.is_half:
                hubert_model = hubert_model.half()
            else:
                hubert_model = hubert_model.float()
            hubert_model.eval()
            self.model = hubert_model
            cpt = torch.load(pth_path, map_location="cpu")
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.version == "v1":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                else:
                    self.net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif self.version == "v2":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                else:
                    self.net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del self.net_g.enc_q
            print(self.net_g.load_state_dict(cpt["weight"], strict=False))
            self.net_g.eval().to(device)
            if config.is_half:
                self.net_g = self.net_g.half()
            else:
                self.net_g = self.net_g.float()
            self.is_half = config.is_half
        except:
            print(traceback.format_exc())

    def _apply_auto_pitch_correction(self, f0, base_f0_up_key):
        if not self.auto_pitch_correction:
            self.last_adjustment = 0.0
            self.pitch_adj_semitones = 0.0
            return base_f0_up_key

        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) < 5:
            self.pitch_adj_semitones = self.last_adjustment
            return base_f0_up_key + self.last_adjustment

        average_f0_hz = np.mean(voiced_f0)
        self.last_avg_hz = average_f0_hz
        
        current_strength = self.auto_pitch_strength
        current_max_adjustment = self.auto_pitch_max_adjustment

        if self.use_split_pitch_correction:
            if average_f0_hz < self.split_pitch_crossover_hz:
                current_strength = self.low_pitch_strength
                current_max_adjustment = self.low_pitch_max_adjustment
            else:
                current_strength = self.high_pitch_strength
                current_max_adjustment = self.high_pitch_max_adjustment

        if self.use_shout_dampening and self.voice_profile in ["Male to Female", "Female to Male"] and average_f0_hz > 250.0:
            shout_threshold_hz = 250.0
            semitones_above_threshold = 12 * np.log2(average_f0_hz / shout_threshold_hz)
            adjustment = -semitones_above_threshold * self.shout_dampening_strength
        else:
            if self.voice_profile == "Male to Female":
                neutral_pitch_hz = 190.0
            elif self.voice_profile == "Female to Male":
                neutral_pitch_hz = 120.0
            else: # Should not happen if auto_pitch_correction is on, but as a safeguard
                self.pitch_adj_semitones = 0.0
                return base_f0_up_key
            
            if average_f0_hz <= 1.0:
                self.pitch_adj_semitones = self.last_adjustment
                return base_f0_up_key + self.last_adjustment

            deviation_semitones = 12 * np.log2(average_f0_hz / neutral_pitch_hz)
            adjustment = -deviation_semitones * current_strength

        max_pitch_change = 0.5
        adjustment = np.clip(adjustment, self.last_adjustment - max_pitch_change, self.last_adjustment + max_pitch_change)
        adjustment = np.clip(adjustment, -current_max_adjustment, current_max_adjustment)
        
        smoothed_adjustment = self.last_adjustment * self.pitch_stability + adjustment * (1 - self.pitch_stability)
        self.last_adjustment = smoothed_adjustment
        self.pitch_adj_semitones = smoothed_adjustment
        
        return base_f0_up_key + smoothed_adjustment

    def _apply_discord_effects(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        # --- Store diagnostics ---
        quality_cutoff_diag = "N/A"
        prox_hp_cutoff_diag = "N/A"
        
        # Quality Effect (Low-pass filter)
        if self.discord_quality < 1.0:
            min_cutoff = 3000
            max_cutoff = min(18000, sr / 2.1)
            cutoff = min_cutoff + self.discord_quality * (max_cutoff - min_cutoff)
            quality_cutoff_diag = f"{cutoff:.0f}Hz"
            if cutoff < sr / 2:
                b, a = signal.butter(4, cutoff, btype='low', fs=sr)
                audio_np = signal.lfilter(b, a, audio_np)

        # Proximity Effect (Volume + High-pass filter for distance)
        volume_scale = 0.4 + self.discord_proximity * 0.6
        audio_np *= volume_scale
        
        if self.discord_proximity < 0.5:
            cutoff = 400 - (self.discord_proximity / 0.5) * 300
            prox_hp_cutoff_diag = f"{cutoff:.0f}Hz"
            if cutoff > 0:
                b, a = signal.butter(2, cutoff, btype='high', fs=sr)
                audio_np = signal.lfilter(b, a, audio_np)

        # Noise Effect (from dB)
        if self.discord_noise > -80.0:
            amplitude = 10 ** (self.discord_noise / 20)
            noise = np.random.randn(len(audio_np)) * amplitude
            audio_np += noise
        
        # Store for CMD display
        self.effects_diag.update({
            "prox_vol": volume_scale, "prox_hp": prox_hp_cutoff_diag,
            "quality_lp": quality_cutoff_diag
        })

        return np.clip(audio_np, -1.0, 1.0)
    
    def _apply_phone_effect(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        # Phone Effect (Band-pass filter and slight distortion)
        b, a = signal.butter(4, [300, 3400], btype='bandpass', fs=sr)
        audio_np = signal.lfilter(b, a, audio_np)
        audio_np = np.tanh(audio_np * 1.2) / 1.2 # Subtle distortion
        return np.clip(audio_np, -1.0, 1.0)

    def _apply_saturation_effect(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        saturation_active_diag = "Off"
        # Pitch-based Saturation
        if self.last_avg_hz > self.saturation_threshold_hz:
            if pedalboard is not None:
                saturation_effect = pedalboard.Distortion(drive_db=self.saturation_drive_db)
                audio_np = saturation_effect(audio_np, sr)
                saturation_active_diag = f"On ({self.saturation_drive_db:.1f}dB)"
        
        self.effects_diag.update({
            "saturation": saturation_active_diag
        })
        return np.clip(audio_np, -1.0, 1.0)

    def _apply_cave_effect(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        if pedalboard is not None:
            # A strong reverb combined with a delay creates the cave effect
            reverb_effect = pedalboard.Reverb(
                room_size=0.8, 
                damping=0.5, 
                wet_level=self.cave_mix, 
                dry_level=1.0 - self.cave_mix
            )
            delay_effect = pedalboard.Delay(
                delay_seconds=self.cave_delay_time / 1000.0, # convert ms to s
                feedback=self.cave_feedback, 
                mix=self.cave_mix
            )
            
            # Apply reverb first, then delay
            audio_np = reverb_effect(audio_np, sr)
            audio_np = delay_effect(audio_np, sr)
        
        return np.clip(audio_np, -1.0, 1.0)

    def _apply_low_freq_dampening(self, audio_np: np.ndarray, sr: int) -> np.ndarray:
        dampening_active_diag = "Off"
        if 0 < self.last_avg_hz < self.low_freq_dampening_threshold_hz:
            reduction_factor = 10 ** (self.low_freq_dampening_level_db / 20)
            audio_np *= reduction_factor
            dampening_active_diag = f"On ({self.low_freq_dampening_level_db:.1f}dB)"

        self.effects_diag.update({
            "low_damp": dampening_active_diag
        })
        return np.clip(audio_np, -1.0, 1.0)

    def _apply_dynamic_proximity(self, audio_np: np.ndarray, sr: int, rms_level: float) -> np.ndarray:
        # Map RMS range (-60 to 0 dB) to a proximity factor (0.0 to 1.0)
        # Use a safe minimum for rms_level to avoid log(0)
        db_level = 20 * np.log10(max(rms_level, 1e-6))
        
        # Normalize dB to a 0-1 range (e.g., -50dB is far, -10dB is close)
        prox_factor = np.interp(db_level, [-50, -10], [0.0, 1.0])
        prox_factor = np.clip(prox_factor, 0.0, 1.0)
        
        self.effects_diag['dyn_prox'] = prox_factor
        
        if pedalboard is not None:
            # Calculate effect parameters based on proximity
            strength = self.dynamic_proximity_strength
            
            # 1. Volume: quieter when far
            volume_factor = 1.0 - (1.0 - prox_factor) * strength * 0.5 # Max 50% reduction
            audio_np *= volume_factor
            
            # 2. Bass (Low Shelf): more bass when close
            bass_boost_db = prox_factor * strength * 6.0 # Max +6dB
            bass_shelf = pedalboard.LowShelfFilter(cutoff_frequency_hz=250, gain_db=bass_boost_db, q=0.7)
            audio_np = bass_shelf(audio_np, sr)
            
            # 3. Treble (High Shelf): more muffled when far
            treble_cut_db = (1.0 - prox_factor) * strength * -9.0 # Max -9dB
            treble_shelf = pedalboard.HighShelfFilter(cutoff_frequency_hz=4000, gain_db=treble_cut_db, q=0.7)
            audio_np = treble_shelf(audio_np, sr)
            
            # 4. Reverb: more room sound when far
            reverb_mix = (1.0 - prox_factor) * strength * 0.5 # Max 50% mix
            room_reverb = pedalboard.Reverb(
                room_size=self.dynamic_proximity_room_size,
                wet_level=reverb_mix,
                dry_level=1.0 - reverb_mix
            )
            audio_np = room_reverb(audio_np, sr)
            
        return np.clip(audio_np, -1.0, 1.0)

    def get_f0_post(self, f0):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)
        return f0_coarse, f0bak

    def get_f0(self, x, f0_up_key, f0method):
        if f0method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        self.last_avg_hz = 0.0
        return None, None

    def get_f0_rmvpe(self, x, f0_up_key):
        if not hasattr(self, "model_rmvpe"):
            from rmvpe import RMVPE
            print("loading rmvpe model")
            self.model_rmvpe = RMVPE("rmvpe.pt", is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio(x, thred=self.rmvpe_threshold)
        
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 5:
            self.last_avg_hz = np.mean(voiced_f0)
        else:
            self.last_avg_hz = 0.0

        final_f0_up_key = self._apply_auto_pitch_correction(f0, f0_up_key)
        f0 *= pow(2, final_f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(self, feats: torch.Tensor, indata: np.ndarray, rate1, rate2, cache_pitch, cache_pitchf, f0method, rms_level: float) -> np.ndarray:
        feats = feats.view(1, -1)
        if config.is_half: feats = feats.half()
        else: feats = feats.float()
        feats = feats.to(self.device)
        
        with torch.no_grad():
            # BUG FIX: Changed _helpers.shape back to feats.shape
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 9 if self.version == "v1" else 12}
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
        
        if self.index is not None and self.index_rate != 0:
            try:
                leng_replace_head = int(rate1 * feats[0].shape[0])
                npy = feats[0][-leng_replace_head:].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                if config.is_half: npy = npy.astype("float16")
                feats[0][-leng_replace_head:] = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate + (1 - self.index_rate) * feats[0][-leng_replace_head:])
            except:
                traceback.print_exc()

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(indata, self.f0_up_key, f0method)
            
            if self.voice_profile != "Default":
                pitchf *= self.formant_shift
                pitchf += (self.timbre - 1.0) * self.sr / self.window

            if pitch.shape[0] > 0:
                cache_pitch = np.roll(cache_pitch, -pitch.shape[0])
                cache_pitch[-pitch.shape[0]:] = pitch
                cache_pitchf = np.roll(cache_pitchf, -pitchf.shape[0])
                cache_pitchf[-pitchf.shape[0]:] = pitchf
            p_len = min(feats.shape[1], 13000, cache_pitch.shape[0])
        else:
            pitch, pitchf = None, None
            p_len = min(feats.shape[1], 13000)
        
        feats = feats[:, :p_len, :]
        if self.if_f0 == 1:
            local_cache_pitch = torch.LongTensor(cache_pitch[:p_len]).unsqueeze(0).to(self.device)
            local_cache_pitchf = torch.FloatTensor(cache_pitchf[:p_len]).unsqueeze(0).to(self.device)
        
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = (self.net_g.infer(feats, p_len_tensor, local_cache_pitch, local_cache_pitchf, sid, rate2)[0][0, 0].data.cpu().float())
            else:
                infered_audio = (self.net_g.infer(feats, p_len_tensor, sid, rate2)[0][0, 0].data.cpu().float())
        
        # Convert to numpy for effects processing
        audio_np = infered_audio.numpy()
        
        # Apply Low Frequency Dampening if enabled
        if self.enable_low_freq_dampening:
            audio_np = self._apply_low_freq_dampening(audio_np, self.tgt_sr)

        # Apply Dynamic Proximity if enabled
        if self.enable_dynamic_proximity:
            audio_np = self._apply_dynamic_proximity(audio_np, self.tgt_sr, rms_level)

        # Apply Discord Effects if enabled
        if self.enable_discord_effects:
            audio_np = self._apply_discord_effects(audio_np, self.tgt_sr)
        
        # Apply Phone Effect if enabled
        if self.enable_phone_effect:
            audio_np = self._apply_phone_effect(audio_np, self.tgt_sr)
        
        # Apply Saturation if enabled
        if self.enable_saturation_effect:
            audio_np = self._apply_saturation_effect(audio_np, self.tgt_sr)

        # Apply Cave Effect if enabled
        if self.enable_cave_effect:
            audio_np = self._apply_cave_effect(audio_np, self.tgt_sr)

        # Apply Reverb if enabled
        if self.reverb_effect is not None:
            audio_np = self.reverb_effect(audio_np, self.tgt_sr)
        
        # Convert back to tensor for final output
        infered_audio = torch.from_numpy(audio_np.astype(np.float32))

        return infered_audio