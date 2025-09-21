# gui_v1.py (COMPLETE AND FIXED)
import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

if __name__ == "__main__":
    import numpy as np
    import traceback, re
    import json, queue
    import PySimpleGUI as sg
    import sounddevice as sd
    import librosa, torch, time, threading, webbrowser
    from rvc_for_realtime import RVC
    from config import Config
    from audio_processor import AudioProcessor

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    config = Config()

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 0
            self.block_time: float = 1.0
            self.threhold: int = -45
            self.crossfade_time: float = 80.0
            self.extra_time: float = 40.0
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.index_rate = 0.75
            self.use_index_file = True
            self.f0method = "rmvpe"
            self.is_half = True
            self.hop_length: int = 128
            self.device_samplerate: int = 48000
            self.sola_search_ms: float = 10.0
            self.input_volume: float = 1.0
            self.input_formant_shift: float = 1.0
            self.input_timbre: float = 1.0
            self.output_volume: float = 0.0
            self.output_volume_normalization: bool = False
            self.bypass_vc: bool = False
            self.voice_profile: str = "Default"
            self.timbre: float = 1.0 
            self.use_split_pitch_correction: bool = False
            self.split_pitch_crossover: str = "C4"
            self.use_reverb: bool = False
            self.reverb_room_size: float = 0.5
            self.reverb_damping: float = 0.5
            self.reverb_wet_level: float = 0.33
            self.reverb_dry_level: float = 0.4
            self.enable_discord_effects: bool = False
            self.discord_proximity: float = 1.0
            self.discord_noise: float = -80.0
            self.discord_quality: float = 1.0
            self.enable_phone_effect: bool = False
            self.enable_saturation_effect: bool = False
            self.saturation_threshold_hz: float = 800.0
            self.saturation_drive_db: float = 6.0
            self.enable_cave_effect: bool = False
            self.cave_delay_time: float = 250.0
            self.cave_feedback: float = 0.4
            self.cave_mix: float = 0.5
            self.enable_low_freq_dampening: bool = False
            self.low_freq_dampening_threshold_hz: float = 100.0
            self.low_freq_dampening_level_db: float = -6.0
            self.enable_dynamic_proximity: bool = False
            self.dynamic_proximity_strength: float = 0.5
            self.dynamic_proximity_room_size: float = 0.2
            self.enable_cmd_diagnostics: bool = False
            self.device = device
            self.input_source: str = "microphone"
            self.input_audio_path: str = ""
            self.loop_audio_file: bool = False
            self.enable_equalizer: bool = False
            self.eq_gains_db = {hz: 0.0 for hz in [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]}
            self.enable_input_equalizer: bool = False
            self.input_eq_gains_db = {hz: 0.0 for hz in [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]}
            
            # Advanced Pitch Shaper Params
            self.f0_smoothing_factor: float = 0.0
            self.auto_pitch_correction: bool = False
            self.pitch_stability: float = 0.2
            self.auto_pitch_strength: float = 0.3
            self.auto_pitch_max_adjustment: float = 2.0
            self.use_shout_dampening: bool = False
            self.shout_dampening_strength: float = 0.8
            self.low_pitch_strength: float = 0.3
            self.low_pitch_max_adjustment: float = 2.0
            self.high_pitch_strength: float = 0.3
            self.high_pitch_max_adjustment: float = 2.0
            self.enable_dynamic_timbre: bool = False
            self.dyn_timbre_attack_ms: float = 100.0
            self.dyn_timbre_release_ms: float = 200.0
            self.low_pitch_crossover: float = 150.0
            self.high_pitch_crossover: float = 400.0
            self.low_timbre_target: float = 1.0
            self.mid_timbre_target: float = 1.0
            self.high_timbre_target: float = 1.0
            self.low_curve_factor: float = 1.0
            self.high_curve_factor: float = 1.0
            self.enable_brightness_sensor: bool = False
            self.brightness_threshold: float = 0.8
            self.min_f0_for_brightness: float = 170.0
            self.formant_shift: float = 1.0
            self.enable_dynamic_formant: bool = False
            self.formant_attack_ms: float = 100.0
            self.formant_release_ms: float = 100.0
            self.low_formant_target: float = 1.0
            self.mid_formant_target: float = 1.0
            self.high_formant_target: float = 1.0
            self.low_formant_curve_factor: float = 1.0
            self.high_formant_curve_factor: float = 1.0

    class GUI:
        def __init__(self) -> None:
            self.config = GUIConfig()
            self.audio_processor: AudioProcessor = None
            self.crossover_notes_list = self.get_crossover_notes()
            self.realtime_param_keys = []
            self.default_settings = {}
            self.eq_freq_bands = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
            self.profiles_folder = os.path.join(now_dir, "weights", "_configsettings")
            os.makedirs(self.profiles_folder, exist_ok=True)
            self.launcher()

        def get_crossover_notes(self):
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            formatted_notes = []
            for octave in range(2, 6):
                for note in note_names:
                    freq = 440.0 * (2.0 ** ((octave - 4) + (note_names.index(note) - 9) / 12.0))
                    note_type = "Low" if octave < 4 else "Mid" if octave == 4 else "High"
                    formatted_string = f"{note}{octave} ({note_type}) - {freq:.2f} Hz"
                    formatted_notes.append(formatted_string)
            return formatted_notes

        def get_note_from_display_string(self, display_string):
            match = re.match(r"([A-G]#?\d+)", display_string)
            return match.group(1) if match else "C4"
            
        def get_display_string_from_note(self, note_to_find):
            for display_string in self.crossover_notes_list:
                if display_string.startswith(note_to_find):
                    return display_string
            return self.crossover_notes_list[0] 

        def get_default_settings(self, input_devices, output_devices):
            return {
                "pth_path": "", "index_path": "",
                "sg_input_device": input_devices[sd.default.device[0]] if sd.default.device[0] < len(input_devices) else '',
                "sg_output_device": output_devices[sd.default.device[1]] if sd.default.device[1] < len(output_devices) else '',
                "threhold": -45, "pitch": 0, "index_rate": 0.75, "use_index_file": True,
                "block_time": 1.0, "crossfade_length": 80.0, "extra_time": 40.0,
                "f0method": "rmvpe", "is_half": True, "hop_length": 128,
                "device_samplerate": 48000, "sola_search_ms": 10.0,
                "I_noise_reduce": False, "O_noise_reduce": False,
                "input_volume": 1.0, "input_formant_shift": 1.0, "input_timbre": 1.0, "output_volume": 0.0, "output_volume_normalization": False, "bypass_vc": False,
                "voice_profile": "Default", "timbre": 1.0,
                "use_split_pitch_correction": False, "split_pitch_crossover": "C4",
                "use_reverb": False, "reverb_room_size": 0.5, "reverb_damping": 0.5,
                "reverb_wet_level": 0.33, "reverb_dry_level": 0.4,
                "enable_discord_effects": False, "discord_proximity": 1.0, 
                "discord_noise": -80.0, "discord_quality": 1.0,
                "enable_phone_effect": False, 
                "enable_saturation_effect": False, "saturation_threshold_hz": 800.0, "saturation_drive_db": 6.0,
                "enable_cave_effect": False, "cave_delay_time": 250.0, "cave_feedback": 0.4, "cave_mix": 0.5,
                "enable_low_freq_dampening": False, "low_freq_dampening_threshold_hz": 100.0, "low_freq_dampening_level_db": -6.0,
                "enable_dynamic_proximity": False, "dynamic_proximity_strength": 0.5, "dynamic_proximity_room_size": 0.2,
                "enable_cmd_diagnostics": False,
                "input_source": "microphone", "input_audio_path": "", "loop_audio_file": False,
                "enable_equalizer": False, "eq_gains_db": {hz: 0.0 for hz in self.eq_freq_bands},
                "enable_input_equalizer": False, "input_eq_gains_db": {hz: 0.0 for hz in self.eq_freq_bands},
                "f0_smoothing_factor": 0.0, "auto_pitch_correction": False, "pitch_stability": 0.2, "auto_pitch_strength": 0.3,
                "auto_pitch_max_adjustment": 2.0, "use_shout_dampening": False, "shout_dampening_strength": 0.8,
                "low_pitch_strength": 0.3, "low_pitch_max_adjustment": 2.0, "high_pitch_strength": 0.3, "high_pitch_max_adjustment": 2.0,
                "enable_dynamic_timbre": False, "dyn_timbre_attack_ms": 100.0, "dyn_timbre_release_ms": 200.0,
                "low_pitch_crossover": 150.0, "high_pitch_crossover": 400.0, "low_timbre_target": 1.0,
                "mid_timbre_target": 1.0, "high_timbre_target": 1.0, "low_curve_factor": 1.0, "high_curve_factor": 1.0,
                "enable_brightness_sensor": False, "brightness_threshold": 0.8, "min_f0_for_brightness": 170.0,
                "formant_shift": 1.0, "enable_dynamic_formant": False, "formant_attack_ms": 100.0, "formant_release_ms": 100.0,
                "low_formant_target": 1.0, "mid_formant_target": 1.0, "high_formant_target": 1.0, 
                "low_formant_curve_factor": 1.0, "high_formant_curve_factor": 1.0
            }

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices(update=True)
            try:
                with open("values1.json", "r") as j:
                    data = json.load(j)
                    default_data = self.get_default_settings(input_devices, output_devices)
                    for key, value in default_data.items(): data.setdefault(key, value)
                    for key in ['eq_gains_db', 'input_eq_gains_db']:
                        if key in data:
                            data[key] = {int(k): float(v) for k, v in data[key].items()}
            except:
                data = self.get_default_settings(input_devices, output_devices)
                with open("values1.json", "w") as j:
                    json.dump(data, j, indent=4)
            return data
        
        def _get_values_for_saving(self, values):
            save_values = values.copy()
            profile = "Default"
            if values.get('m2f'): profile = "Male to Female"
            elif values.get('f2m'): profile = "Female to Male"
            elif values.get('f2f'): profile = "Female to Female"
            elif values.get('m2m'): profile = "Male to Male"
            save_values['voice_profile'] = profile
            
            save_values['input_source'] = "file" if values.get('file_input') else "microphone"
            save_values["split_pitch_crossover"] = self.get_note_from_display_string(values.get("split_pitch_crossover_display", "C4"))
            
            save_values['eq_gains_db'] = {hz: values.get(f'eq_gain_{hz}hz_db', 0.0) for hz in self.eq_freq_bands}
            save_values['input_eq_gains_db'] = {hz: values.get(f'input_eq_gain_{hz}hz_db', 0.0) for hz in self.eq_freq_bands}
            
            keys_to_remove = [f'eq_gain_{hz}hz_db' for hz in self.eq_freq_bands] + \
                             [f'input_eq_gain_{hz}hz_db' for hz in self.eq_freq_bands] + \
                             ["split_pitch_crossover_display", "-EQ_PRESET_COMBO-", 'm2f', 'f2m', 'f2f', 'm2m', 'default', 'mic_input', 'file_input']
            
            final_values = {k: v for k, v in save_values.items() if k not in keys_to_remove and not isinstance(v, (list, tuple, dict)) or k in ['eq_gains_db', 'input_eq_gains_db']}
            return final_values

        def save(self, values):
            temp_values = self._get_values_for_saving(values)
            with open("values1.json", "w") as j:
                json.dump(temp_values, j, indent=4)
            print("Settings saved to values1.json")
        
        def _save_profile(self, values, profile_name):
            if not profile_name.strip():
                sg.popup("Profile name cannot be empty.", title="Error")
                return
            save_data = self._get_values_for_saving(values)
            profile_path = os.path.join(self.profiles_folder, f"{profile_name}.json")
            with open(profile_path, "w") as j:
                json.dump(save_data, j, indent=4)
            sg.popup(f"Profile '{profile_name}' saved successfully.", title="Success")
            self._update_profile_list()

        def _load_profile(self, profile_name):
            profile_path = os.path.join(self.profiles_folder, f"{profile_name}.json")
            try:
                with open(profile_path, "r") as j:
                    data = json.load(j)
                self._apply_settings_to_gui_and_backend(data)
                sg.popup(f"Profile '{profile_name}' loaded successfully.", title="Success")
            except Exception as e:
                sg.popup(f"Error loading profile '{profile_name}': {e}", title="Error")

        def _delete_profile(self, profile_name):
            profile_path = os.path.join(self.profiles_folder, f"{profile_name}.json")
            if os.path.exists(profile_path):
                confirm = sg.popup_yes_no(f"Are you sure you want to delete the profile '{profile_name}'?", title="Confirm Deletion")
                if confirm == 'Yes':
                    os.remove(profile_path)
                    sg.popup(f"Profile '{profile_name}' deleted.", title="Success")
                    self._update_profile_list()
            else:
                sg.popup(f"Profile '{profile_name}' not found.", title="Error")
                
        def _update_profile_list(self):
            try:
                profiles = [f.replace(".json", "") for f in os.listdir(self.profiles_folder) if f.endswith(".json")]
                self.window['-PROFILE_LIST-'].update(values=profiles, value=profiles[0] if profiles else "")
            except Exception as e:
                print(f"Error updating profile list: {e}")

        def _apply_settings_to_gui_and_backend(self, data):
            for key, value in data.items():
                if key in self.window.key_dict:
                    element = self.window[key]
                    if isinstance(element, (sg.Slider, sg.Input, sg.Checkbox)):
                        element.update(value)
                    elif isinstance(element, sg.Combo):
                        if value in element.Values or key in ['sg_input_device', 'sg_output_device']:
                            element.update(value)
            
            profile = data.get("voice_profile", "Default")
            profile_map = {"Default": "default", "Male to Female": "m2f", "Female to Male": "f2m", "Female to Female": "f2f", "Male to Male": "m2m"}
            for p_name, p_key in profile_map.items():
                self.window[p_key].update(profile == p_name)
            self.update_pitch_shaper_visibility(profile)

            source = data.get("input_source", "microphone")
            self.window['mic_input'].update(source == 'microphone')
            self.window['file_input'].update(source == 'file')
            self.window['audio_file_row'].update(visible=source == 'file')
            self.window['input_device_col'].update(visible=source == 'microphone')

            for prefix in ["", "input_"]:
                gains_key = f"{prefix}eq_gains_db"
                if gains_key in data:
                    gains = {int(k): float(v) for k, v in data[gains_key].items()}
                    for hz, gain in gains.items():
                        self.window[f'{prefix}eq_gain_{hz}hz_db'].update(gain)
                        self.window[f'{prefix}eq_gain_label_{hz}hz'].update(f"{gain:.0f} dB")

            crossover_note = data.get("split_pitch_crossover", "C4")
            self.window["split_pitch_crossover_display"].update(self.get_display_string_from_note(crossover_note))
            
            for vis_key in self.visibility_param_keys:
                if vis_key in data:
                    self.update_visibility(vis_key, data[vis_key])
            
            if self.audio_processor:
                params_to_update = data.copy()
                params_to_update["voice_profile"] = profile
                params_to_update["split_pitch_crossover"] = crossover_note
                params_to_update[f'input_eq_gains_db'] = {int(k): v for k, v in data.get('input_eq_gains_db', {}).items()}
                params_to_update[f'eq_gains_db'] = {int(k): v for k, v in data.get('eq_gains_db', {}).items()}
                
                for key, value in params_to_update.items():
                    if hasattr(self.config, key) or hasattr(self.audio_processor.rvc, key):
                        self.audio_processor.queue_parameter_update(key, value)


        def launcher(self):
            data = self.load()
            sg.theme("DarkGrey9")
            input_devices, output_devices, _, _ = self.get_devices(update=False)
            self.default_settings = self.get_default_settings(input_devices, output_devices)
            samplerate_options = [16000, 32000, 40000, 44100, 48000, 96000]
            
            crossover_notes = self.get_crossover_notes()
            default_display_crossover = self.get_display_string_from_note(data.get("split_pitch_crossover", "C4"))

            current_profile = data.get("voice_profile", "Default")
            enable_dynamic_formant = data.get("enable_dynamic_formant", False)
            enable_dynamic_timbre = data.get("enable_dynamic_timbre", False)
            use_index_file = data.get("use_index_file", True)
            auto_pitch_correction_enabled = data.get("auto_pitch_correction", False)
            enable_equalizer = data.get("enable_equalizer", False)
            enable_input_equalizer = data.get("enable_input_equalizer", False)
            output_volume_normalization_enabled = data.get("output_volume_normalization", False)
            bypass_vc_enabled = data.get("bypass_vc", False)
            use_shout_dampening = data.get("use_shout_dampening", False)
            use_split_pitch_correction = data.get("use_split_pitch_correction", False)
            use_reverb = data.get("use_reverb", False)
            enable_discord_effects = data.get("enable_discord_effects", False)
            enable_phone_effect = data.get("enable_phone_effect", False)
            enable_saturation_effect = data.get("enable_saturation_effect", False)
            enable_cave_effect = data.get("enable_cave_effect", False)
            enable_low_freq_dampening = data.get("enable_low_freq_dampening", False)
            enable_dynamic_proximity = data.get("enable_dynamic_proximity", False)
            current_input_source = data.get("input_source", "microphone")

            self.realtime_param_keys = [
                "threhold", "pitch", "index_rate", "input_volume", "input_formant_shift", "input_timbre", "output_volume", "I_noise_reduce", "O_noise_reduce", "timbre", "bypass_vc",
                "f0_smoothing_factor", "auto_pitch_correction", "pitch_stability", "auto_pitch_strength", "auto_pitch_max_adjustment", "use_shout_dampening", "shout_dampening_strength",
                "use_split_pitch_correction", "low_pitch_strength", "low_pitch_max_adjustment", "high_pitch_strength", "high_pitch_max_adjustment", "split_pitch_crossover_display",
                "enable_dynamic_formant", "formant_shift", "formant_attack_ms", "formant_release_ms", "low_formant_target", "mid_formant_target", "high_formant_target", "low_formant_curve_factor", "high_formant_curve_factor",
                "enable_dynamic_timbre", "dyn_timbre_attack_ms", "dyn_timbre_release_ms", "low_pitch_crossover", "high_pitch_crossover", "low_timbre_target", "mid_timbre_target", "high_timbre_target", "low_curve_factor", "high_curve_factor",
                "enable_brightness_sensor", "brightness_threshold", "min_f0_for_brightness",
                "use_reverb", "reverb_room_size", "reverb_damping", "reverb_wet_level", "reverb_dry_level",
                "enable_discord_effects", "discord_proximity", "discord_noise", "discord_quality",
                "enable_phone_effect", "enable_saturation_effect", "saturation_threshold_hz", "saturation_drive_db",
                "enable_cave_effect", "cave_delay_time", "cave_feedback", "cave_mix",
                "enable_low_freq_dampening", "low_freq_dampening_threshold_hz", "low_freq_dampening_level_db",
                "enable_dynamic_proximity", "dynamic_proximity_strength", "dynamic_proximity_room_size", "enable_cmd_diagnostics",
                "enable_equalizer", "output_volume_normalization", "enable_input_equalizer"
            ] + [f'eq_gain_{hz}hz_db' for hz in self.eq_freq_bands] + [f'input_eq_gain_{hz}hz_db' for hz in self.eq_freq_bands]

            self.visibility_param_keys = ["use_shout_dampening", "use_split_pitch_correction", "use_reverb", "enable_discord_effects", "enable_saturation_effect", "enable_cave_effect", "enable_low_freq_dampening", "enable_dynamic_proximity", "auto_pitch_correction", "enable_phone_effect", "enable_dynamic_timbre", "enable_dynamic_formant", "enable_equalizer", "enable_input_equalizer"]
            
            self.main_reset_keys = ["threhold", "pitch", "index_rate", "input_volume", "input_formant_shift", "input_timbre", "output_volume", "output_volume_normalization", "bypass_vc", "default", "m2f", "f2m", "f2f", "m2m", "enable_input_equalizer", "input_eq_gains_db"]
            self.effects_reset_keys = ["timbre", "enable_dynamic_timbre", "dyn_timbre_attack_ms", "dyn_timbre_release_ms", "low_pitch_crossover", "high_pitch_crossover", "low_timbre_target", "mid_timbre_target", "high_timbre_target", "low_curve_factor", "high_curve_factor", "enable_brightness_sensor", "brightness_threshold", "min_f0_for_brightness", "enable_dynamic_formant", "formant_shift", "formant_attack_ms", "formant_release_ms", "low_formant_target", "mid_formant_target", "high_formant_target", "low_formant_curve_factor", "high_formant_curve_factor", "enable_low_freq_dampening", "low_freq_dampening_threshold_hz", "low_freq_dampening_level_db", "enable_saturation_effect", "saturation_threshold_hz", "saturation_drive_db", "enable_dynamic_proximity", "dynamic_proximity_strength", "dynamic_proximity_room_size", "enable_discord_effects", "discord_proximity", "discord_noise", "discord_quality", "enable_cave_effect", "cave_delay_time", "cave_feedback", "cave_mix", "use_reverb", "reverb_room_size", "reverb_damping", "reverb_wet_level", "reverb_dry_level", "enable_phone_effect"]
            self.autopitch_reset_keys = ["f0_smoothing_factor", "auto_pitch_correction", "auto_pitch_strength", "auto_pitch_max_adjustment", "pitch_stability", "use_shout_dampening", "shout_dampening_strength", "use_split_pitch_correction", "split_pitch_crossover_display", "low_pitch_strength", "low_pitch_max_adjustment", "high_pitch_strength", "high_pitch_max_adjustment"]
            self.performance_reset_keys = ["I_noise_reduce", "O_noise_reduce", "enable_cmd_diagnostics"]
            self.equalizer_reset_keys = ["enable_equalizer", "eq_gains_db"]
            
            input_eq_band_columns = []
            input_eq_gains = data.get('input_eq_gains_db', {hz: 0.0 for hz in self.eq_freq_bands})
            for hz in self.eq_freq_bands:
                gain_val = input_eq_gains.get(str(hz), input_eq_gains.get(hz, 0.0))
                freq_label = f"{hz} Hz" if hz < 1000 else f"{hz/1000:.1f}k".replace(".0","")
                band_col = sg.Column([
                    [sg.Text(freq_label, size=(6,1), justification='center', font=("Helvetica", 9))],
                    [sg.Slider(range=(12, -12), key=f'input_eq_gain_{hz}hz_db', resolution=1, orientation="v", size=(4, 18), default_value=gain_val, enable_events=True, disable_number_display=True)],
                    [sg.Text(f"{gain_val:.0f} dB", key=f"input_eq_gain_label_{hz}hz", size=(6,1), justification='center', font=("Helvetica", 9))]
                ], element_justification='center', pad=(8, 10))
                input_eq_band_columns.append(band_col)
            
            input_equalizer_controls_layout = sg.Column([
                [sg.Button("Flatten", key='-INPUT_EQ_FLATTEN-', size=(11,1)), sg.Button("Invert", key='-INPUT_EQ_INVERT-', size=(11,1))],
                [sg.Button("Bass Boost", key='-INPUT_EQ_BASS_BOOST-', size=(11,1)), sg.Button("Treble Boost", key='-INPUT_EQ_TREBLE_BOOST-', size=(11,1))],
                [sg.Button("Bass Cut", key='-INPUT_EQ_BASS_CUT-', size=(11,1)), sg.Button("Treble Cut", key='-INPUT_EQ_TREBLE_CUT-', size=(11,1))]
            ], pad=(10,10))

            input_equalizer_frame = sg.Frame("Input Equalizer", [
                [sg.Checkbox("Enable Input EQ", key="enable_input_equalizer", default=enable_input_equalizer, enable_events=True, tooltip="Enables a 10-band equalizer for the input audio before it enters the model.")],
                [sg.pin(sg.Column([
                    [input_equalizer_controls_layout],
                    [sg.Frame("10-Band Input Graphic Equalizer", [input_eq_band_columns], border_width=0, pad=(0,0), element_justification='c')]
                ], key='input_equalizer_column', visible=enable_input_equalizer))]
            ], expand_x=True)
            
            main_tab_layout = [
                [sg.Push(), sg.Button("Reset", key="-RESET_MAIN-", size=(8,1), tooltip="Resets all settings on this tab to their default values.")],
                [sg.Text("Response Threshold"), sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -30), expand_x=True, enable_events=True, tooltip="The volume level (in dB) your voice must exceed to be processed.\nEffectively a noise gate. Higher values are stricter.")],
                [sg.Text("Pitch Setting"), sg.Slider(range=(-24, 24), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), expand_x=True, enable_events=True, tooltip="Global pitch shift in semitones (half-steps).\nExample: +12 for one octave up, -12 for one octave down.")],
                [sg.pin(sg.Column([[sg.Text("Index Rate"), sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0.3), expand_x=True, enable_events=True, tooltip="The degree to which the model's timbre is replaced by the feature vector from the .index file.\nHigher values = more of the index file's timbre.") ]], key='index_rate_row', visible=use_index_file))],
                [sg.Text("Input Volume"), sg.Slider(range=(0.0, 2.0), key="input_volume", resolution=0.01, orientation="h", default_value=data.get("input_volume", 1.0), expand_x=True, enable_events=True, tooltip="Adjusts the volume of the input (microphone) signal before processing.\n1.0 is neutral.")],
                [sg.Text("Input Formant"), sg.Slider(range=(0.0, 2.0), key="input_formant_shift", resolution=0.01, orientation="h", default_value=data.get("input_formant_shift", 1.0), expand_x=True, enable_events=True, tooltip="Shifts the formants of the input audio. Approximated by pitch shifting.\nCan help with gender conversion before the main RVC model.")],
                [sg.Text("Input Timbre"), sg.Slider(range=(0.0, 2.0), key="input_timbre", resolution=0.01, orientation="h", default_value=data.get("input_timbre", 1.0), expand_x=True, enable_events=True, tooltip="Adjusts the timbre (brightness) of the input audio via a high-shelf filter.\n> 1.0 is brighter, < 1.0 is darker.")],
                [
                    sg.Text("Output Volume (dB)"), 
                    sg.Slider(range=(-60, 12), key="output_volume", resolution=1, orientation="h", default_value=data.get("output_volume", 0.0), expand_x=True, enable_events=True, disabled=output_volume_normalization_enabled, tooltip="Final output volume adjustment in decibels."),
                    sg.Checkbox("Normalize", key="output_volume_normalization", default=output_volume_normalization_enabled, tooltip="Automatically adjusts volume to a standard level (-14 LUFS).\nDisables the manual Output Volume slider.", enable_events=True)
                ],
                [sg.Checkbox("Bypass Voice Changer", key="bypass_vc", default=bypass_vc_enabled, enable_events=True, tooltip="If checked, passes your microphone audio directly to the output without RVC processing.")],
                [sg.Frame("Voice Profile", [[
                    sg.Radio("Default", "voice_profile", key="default", default=current_profile == "Default", enable_events=True, tooltip="Standard profile with no special pitch correction logic."),
                    sg.Radio("M->F", "voice_profile", key="m2f", default=current_profile == "Male to Female", enable_events=True, tooltip="Male to Female profile. Enables Auto Pitch Correction and other advanced shaping tools."), 
                    sg.Radio("F->M", "voice_profile", key="f2m", default=current_profile == "Female to Male", enable_events=True, tooltip="Female to Male profile. Enables Auto Pitch Correction and other advanced shaping tools."), 
                    sg.Radio("F->F", "voice_profile", key="f2f", default=current_profile == "Female to Female", enable_events=True, tooltip="Female to Female profile. Advanced shaping tools are hidden."), 
                    sg.Radio("M->M", "voice_profile", key="m2m", default=current_profile == "Male to Male", enable_events=True, tooltip="Male to Male profile. Advanced shaping tools are hidden.")
                ]], expand_x=True)],
                [sg.HSep()],
                [input_equalizer_frame]
            ]
            
            model_layout = [
                [sg.Checkbox("Use Index File", key="use_index_file", default=use_index_file, enable_events=True, tooltip="Enables the use of a .index file to help preserve the timbre of the trained voice.")],
                [sg.Input(default_text=data.get("pth_path", ""), key="pth_path", expand_x=True), sg.FileBrowse("Select Model (.pth)", initial_folder=os.path.join(os.getcwd(), "weights"), file_types=((". pth"),), tooltip="Select the main model file for the voice conversion.")],
                [sg.pin(sg.Column([[sg.Input(default_text=data.get("index_path", ""), key="index_path", expand_x=True), sg.FileBrowse("Select Index (.index)", initial_folder=os.path.join(os.getcwd(), "logs"), file_types=((". index"),), tooltip="Select the .index file associated with your model.") ]], key='index_path_row', visible=use_index_file))]
            ]
            
            input_source_layout = [
                [sg.Radio("Microphone", "input_source", key="mic_input", default=current_input_source == "microphone", enable_events=True),
                 sg.Radio("Audio File", "input_source", key="file_input", default=current_input_source == "file", enable_events=True)],
                [sg.pin(sg.Column([[
                    sg.Input(default_text=data.get("input_audio_path", ""), key="input_audio_path", expand_x=True, enable_events=True), 
                    sg.FileBrowse("Select Audio File", file_types=(("Audio Files", "*.wav *.mp3 *.flac"),), tooltip="Select an audio file to use as input instead of a microphone.")
                ]], key='audio_file_row', visible=current_input_source == "file"))]
            ]
            
            devices_layout = [
                [sg.pin(sg.Column([[
                    sg.Text("Input Device"), sg.Combo(input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), expand_x=True, tooltip="Select your microphone.")
                ]], key='input_device_col', visible=current_input_source == "microphone"))],
                [sg.Text("Output Device"), sg.Combo(output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), expand_x=True, tooltip="Select your speakers or virtual audio cable.", enable_events=True)],
                [sg.Text("Device Sample Rate"), sg.Combo(samplerate_options, key="device_samplerate", default_value=data.get("device_samplerate", 48000), expand_x=True, tooltip="The sample rate for your audio devices. Must match your device settings.\n48000Hz is recommended."), sg.Button("Refresh", key="refresh_devices", tooltip="Rescans your system for available audio devices.")]
            ]
            
            model_devices_tab_layout = [[sg.Frame("Load Model", model_layout, expand_x=True)], [sg.Frame("Input Source", input_source_layout, expand_x=True)], [sg.Frame("Audio Devices", devices_layout, expand_x=True)]]
            
            brightness_sensor_layout = sg.Frame("Brightness Sensor", [
                [sg.Checkbox("Enable Brightness Sensor", key="enable_brightness_sensor", default=data.get("enable_brightness_sensor", False), enable_events=True, tooltip="Triggers the High Pitch Band based on spectral brightness (sibilance, shouting) instead of just F0.\nUseful if your high notes don't have a high F0.")],
                [sg.Text("Brightness Threshold"), sg.Slider(range=(0.1, 5.0), key="brightness_threshold", resolution=0.05, orientation="h", default_value=data.get("brightness_threshold", 0.8), expand_x=True, enable_events=True, tooltip="How bright the sound needs to be to trigger the effect.\nLower values are more sensitive.")],
                [sg.Text("Minimum F0 (Hz)"), sg.Slider(range=(80, 300), key="min_f0_for_brightness", resolution=1, orientation="h", default_value=data.get("min_f0_for_brightness", 170.0), expand_x=True, enable_events=True, tooltip="The Brightness Sensor will only activate if your F0 is above this value.\nPrevents 'sss' sounds on low notes from triggering the effect.")],
            ], expand_x=True)

            dynamic_timbre_settings_layout = [
                [sg.Text("Attack (ms)"), sg.Slider(range=(10, 500), key="dyn_timbre_attack_ms", resolution=10, orientation="h", default_value=data.get("dyn_timbre_attack_ms", 100.0), expand_x=True, enable_events=True, tooltip="How quickly the timbre adapts to changes (in milliseconds).")],
                [sg.Text("Release (ms)"), sg.Slider(range=(10, 1000), key="dyn_timbre_release_ms", resolution=10, orientation="h", default_value=data.get("dyn_timbre_release_ms", 200.0), expand_x=True, enable_events=True, tooltip="How quickly the timbre returns to normal (in milliseconds).")],
                [sg.HSep()],
                [sg.Text("Low/Mid Crossover (Hz)"), sg.Slider(range=(50, 300), key="low_pitch_crossover", resolution=1, orientation="h", default_value=data.get("low_pitch_crossover", 150.0), expand_x=True, enable_events=True, tooltip="The F0 frequency that defines the boundary between the Low and Mid pitch bands.")],
                [sg.Text("Mid/High Crossover (Hz)"), sg.Slider(range=(150, 800), key="high_pitch_crossover", resolution=1, orientation="h", default_value=data.get("high_pitch_crossover", 400.0), expand_x=True, enable_events=True, tooltip="The F0 frequency that defines the boundary between the Mid and High pitch bands.")],
                [sg.HSep()],
                [sg.Frame("Low Pitch Band", [
                    [sg.Text("Target Timbre"), sg.Slider(range=(0.0, 2.0), key="low_timbre_target", resolution=0.01, orientation="h", default_value=data.get("low_timbre_target", 1.0), expand_x=True, enable_events=True, tooltip="The target timbre when your voice is in the low pitch band.\nExample: Set > 1.0 for a richer bass.")],
                    [sg.Text("Curve Factor"), sg.Slider(range=(0.1, 3.0), key="low_curve_factor", resolution=0.01, orientation="h", default_value=data.get("low_curve_factor", 1.0), expand_x=True, enable_events=True, tooltip="Controls the transition curve into the low band.\n> 1.0: Slow transition, < 1.0: Fast transition.")],
                ], expand_x=True)],
                [sg.Frame("Mid Pitch Band", [
                    [sg.Text("Target Timbre"), sg.Slider(range=(0.0, 2.0), key="mid_timbre_target", resolution=0.01, orientation="h", default_value=data.get("mid_timbre_target", 1.0), expand_x=True, enable_events=True, tooltip="The baseline timbre for the middle (normal) pitch range.")],
                ], expand_x=True)],
                [sg.Frame("High Pitch Band", [
                    [sg.Text("Target Timbre"), sg.Slider(range=(0.0, 2.0), key="high_timbre_target", resolution=0.01, orientation="h", default_value=data.get("high_timbre_target", 1.0), expand_x=True, enable_events=True, tooltip="The target timbre when your voice is in the high pitch band.\nExample: Set < 1.0 to make high notes less sharp.")],
                    [sg.Text("Curve Factor"), sg.Slider(range=(0.1, 3.0), key="high_curve_factor", resolution=0.01, orientation="h", default_value=data.get("high_curve_factor", 1.0), expand_x=True, enable_events=True, tooltip="Controls the transition curve into the high band.\n> 1.0: Slow start, aggressive at peaks. Good for taming shouts.")],
                ], expand_x=True)],
                [sg.HSep()],
                [brightness_sensor_layout],
            ]
            dynamic_timbre_frame = sg.Frame("Dynamic Timbre Shaper", [
                [sg.Checkbox("Enable Dynamic Timbre", key="enable_dynamic_timbre", default=enable_dynamic_timbre, enable_events=True, tooltip="Enables pitch-aware, multi-band timbre shaping.")],
                [sg.pin(sg.Column(dynamic_timbre_settings_layout, key='dynamic_timbre_column', visible=enable_dynamic_timbre))]
            ], expand_x=True, key="dynamic_timbre_frame")

            dynamic_formant_settings_layout = [
                [sg.Text("Attack (ms)"), sg.Slider(range=(10, 500), key="formant_attack_ms", resolution=10, orientation="h", default_value=data.get("formant_attack_ms", 100.0), expand_x=True, enable_events=True, tooltip="How quickly the formant adapts to changes (in milliseconds). \nHigher values can reduce artifacts.")],
                [sg.Text("Release (ms)"), sg.Slider(range=(10, 500), key="formant_release_ms", resolution=10, orientation="h", default_value=data.get("formant_release_ms", 100.0), expand_x=True, enable_events=True, tooltip="How quickly the formant returns to normal (in milliseconds).")],
                [sg.HSep()],
                [sg.Frame("Low Pitch Band", [
                    [sg.Text("Target Formant"), sg.Slider(range=(0.5, 1.5), key="low_formant_target", resolution=0.01, orientation="h", default_value=data.get("low_formant_target", 1.0), expand_x=True, enable_events=True, tooltip="Target formant shift for the low pitch band.\nExample: < 1.0 to make low notes sound 'bigger'.")],
                    [sg.Text("Curve Factor"), sg.Slider(range=(0.1, 3.0), key="low_formant_curve_factor", resolution=0.01, orientation="h", default_value=data.get("low_formant_curve_factor", 1.0), expand_x=True, enable_events=True, tooltip="Controls the transition curve into the low band.")],
                ], expand_x=True)],
                [sg.Frame("Mid Pitch Band", [
                    [sg.Text("Target Formant"), sg.Slider(range=(0.5, 1.5), key="mid_formant_target", resolution=0.01, orientation="h", default_value=data.get("mid_formant_target", 1.0), expand_x=True, enable_events=True, tooltip="Baseline formant shift for the middle (normal) pitch range.")],
                ], expand_x=True)],
                [sg.Frame("High Pitch Band", [
                    [sg.Text("Target Formant"), sg.Slider(range=(0.5, 1.5), key="high_formant_target", resolution=0.01, orientation="h", default_value=data.get("high_formant_target", 1.0), expand_x=True, enable_events=True, tooltip="Target formant shift for the high pitch band.\nExample: < 1.0 to prevent 'chipmunk' sound on high notes.")],
                    [sg.Text("Curve Factor"), sg.Slider(range=(0.1, 3.0), key="high_formant_curve_factor", resolution=0.01, orientation="h", default_value=data.get("high_formant_curve_factor", 1.0), expand_x=True, enable_events=True, tooltip="Controls the transition curve into the high band.\n> 1.0 = Slower transition, helps prevent artifacts.")],
                ], expand_x=True)],
            ]
            dynamic_formant_frame = sg.Frame("Dynamic Formant Shaper", [
                [sg.Checkbox("Enable Dynamic Formant", key="enable_dynamic_formant", default=enable_dynamic_formant, enable_events=True, tooltip="Enables pitch-aware, multi-band formant shaping for realism.")],
                [sg.pin(sg.Column([[sg.Text("Static Formant Shift"), sg.Slider(range=(0.5, 1.5), key="formant_shift", resolution=0.01, orientation="h", default_value=data.get("formant_shift", 1.0), expand_x=True, enable_events=True, tooltip="A single, global formant shift value.")]], key='static_formant_row', visible=not enable_dynamic_formant))],
                [sg.pin(sg.Column(dynamic_formant_settings_layout, key='dynamic_formant_column', visible=enable_dynamic_formant))]
            ], expand_x=True, key="dynamic_formant_frame")
            
            default_timbre_frame = sg.Frame("Global Timbre", [
                [sg.pin(sg.Column([[sg.Text("Default Timbre"), sg.Slider(range=(0.0, 2.0), key="timbre", resolution=0.01, orientation="h", default_value=data.get("timbre", 1.0), expand_x=True, enable_events=True, tooltip="A single, global timbre adjustment.")]], key='default_timbre_row', visible=not enable_dynamic_timbre))]
            ], expand_x=True)
            
            low_freq_dampening_frame = sg.Frame("Low Freq Dampening", [[sg.Checkbox("Enable", key="enable_low_freq_dampening", default=enable_low_freq_dampening, enable_events=True, tooltip="Reduces low-frequency energy when your pitch is below a threshold.")], [sg.pin(sg.Column([[sg.Text("Threshold (Hz)"), sg.Slider(range=(50, 200), key="low_freq_dampening_threshold_hz", resolution=1, orientation="h", default_value=data.get("low_freq_dampening_threshold_hz", 100.0), expand_x=True, enable_events=True)], [sg.Text("Dampen (dB)"), sg.Slider(range=(-24.0, 0.0), key="low_freq_dampening_level_db", resolution=0.5, orientation="h", default_value=data.get("low_freq_dampening_level_db", -6.0), expand_x=True, enable_events=True)],], key='low_freq_dampening_column', visible=enable_low_freq_dampening))]], expand_x=True)
            saturation_frame = sg.Frame("Saturation", [[sg.Checkbox("Enable", key="enable_saturation_effect", default=enable_saturation_effect, enable_events=True, tooltip="Adds harmonic distortion (drive) when pitch is above a threshold.")], [sg.pin(sg.Column([[sg.Text("Threshold (Hz)"), sg.Slider(range=(100, 2000), key="saturation_threshold_hz", resolution=10, orientation="h", default_value=data.get("saturation_threshold_hz", 800.0), expand_x=True, enable_events=True)], [sg.Text("Drive (dB)"), sg.Slider(range=(0.0, 24.0), key="saturation_drive_db", resolution=0.5, orientation="h", default_value=data.get("saturation_drive_db", 6.0), expand_x=True, enable_events=True)],], key='saturation_column', visible=enable_saturation_effect))]], expand_x=True)
            dynamic_proximity_frame = sg.Frame("Dynamic Proximity", [[sg.Checkbox("Enable", key="enable_dynamic_proximity", default=enable_dynamic_proximity, enable_events=True, tooltip="Simulates moving closer or further from the microphone based on volume.")], [sg.pin(sg.Column([[sg.Text("Strength"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_strength", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_strength", 0.5), expand_x=True, enable_events=True)], [sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_room_size", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_room_size", 0.2), expand_x=True, enable_events=True)],], key='dynamic_proximity_column', visible=enable_dynamic_proximity))]], expand_x=True)
            discord_effects_frame = sg.Frame("Discord Effects", [[sg.Checkbox("Enable", key="enable_discord_effects", default=enable_discord_effects, enable_events=True, tooltip="Simulates various effects common in voice chat applications.")], [sg.pin(sg.Column([[sg.Text("Proximity"), sg.Slider(range=(0.0, 1.0), key="discord_proximity", resolution=0.01, orientation="h", default_value=data.get("discord_proximity", 1.0), expand_x=True, enable_events=True, tooltip="Simulates proximity to a low-quality mic.")], [sg.Text("Noise (dB)"), sg.Slider(range=(-80.0, 12.0), key="discord_noise", resolution=0.5, orientation="h", default_value=data.get("discord_noise", -80.0), expand_x=True, enable_events=True, tooltip="Adds white noise.")], [sg.Text("Quality"), sg.Slider(range=(0.0, 1.0), key="discord_quality", resolution=0.01, orientation="h", default_value=data.get("discord_quality", 1.0), expand_x=True, enable_events=True, tooltip="Simulates audio quality/bitrate reduction.")],], key='discord_effects_column', visible=enable_discord_effects))]], expand_x=True)
            cave_effect_frame = sg.Frame("Cave/Large Room Echo", [[sg.Checkbox("Enable", key="enable_cave_effect", default=enable_cave_effect, enable_events=True, tooltip="A delay/echo effect that simulates a large, reverberant space.")], [sg.pin(sg.Column([[sg.Text("Delay (ms)"), sg.Slider(range=(50, 1000), key="cave_delay_time", resolution=10, orientation="h", default_value=data.get("cave_delay_time", 250.0), expand_x=True, enable_events=True)], [sg.Text("Feedback"), sg.Slider(range=(0.0, 0.9), key="cave_feedback", resolution=0.01, orientation="h", default_value=data.get("cave_feedback", 0.4), expand_x=True, enable_events=True)], [sg.Text("Mix"), sg.Slider(range=(0.0, 1.0), key="cave_mix", resolution=0.01, orientation="h", default_value=data.get("cave_mix", 0.5), expand_x=True, enable_events=True)],], key='cave_effect_column', visible=enable_cave_effect))]], expand_x=True)
            reverb_frame = sg.Frame("Reverb", [[sg.Checkbox("Enable", key="use_reverb", default=use_reverb, enable_events=True, tooltip="Adds reverb to the output signal.")], [sg.pin(sg.Column([[sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="reverb_room_size", resolution=0.01, orientation="h", default_value=data.get("reverb_room_size", 0.5), expand_x=True, enable_events=True)], [sg.Text("Damping"), sg.Slider(range=(0.0, 1.0), key="reverb_damping", resolution=0.01, orientation="h", default_value=data.get("reverb_damping", 0.5), expand_x=True, enable_events=True)], [sg.Text("Wet Level"), sg.Slider(range=(0.0, 1.0), key="reverb_wet_level", resolution=0.01, orientation="h", default_value=data.get("reverb_wet_level", 0.33), expand_x=True, enable_events=True, tooltip="Volume of the 'wet' (reverberated) signal.")], [sg.Text("Dry Level"), sg.Slider(range=(0.0, 1.0), key="reverb_dry_level", resolution=0.01, orientation="h", default_value=data.get("reverb_dry_level", 0.4), expand_x=True, enable_events=True, tooltip="Volume of the 'dry' (original) signal.")],], key='reverb_settings_column', visible=use_reverb))]], expand_x=True)
            phone_effect_frame = sg.Frame("Phone Effect", [[sg.Checkbox("Enable", key="enable_phone_effect", default=enable_phone_effect, enable_events=True, tooltip="Simulates the sound of a telephone call (band-pass filter and saturation).")]], expand_x=True)
            
            effects_col_layout = [
                [dynamic_formant_frame],
                [default_timbre_frame],
                [sg.pin(sg.Column([
                    [dynamic_timbre_frame],
                ], key="dynamic_timbre_master_column"))],
                [dynamic_proximity_frame], 
                [low_freq_dampening_frame], 
                [saturation_frame], 
                [discord_effects_frame], 
                [cave_effect_frame], 
                [reverb_frame], 
                [phone_effect_frame], 
                [sg.Frame('', [[]], size=(10, 800), border_width=0, pad=(0,0))]
            ]
            effects_tab_layout = [[sg.Push(), sg.Button("Reset", key="-RESET_EFFECTS-", size=(8,1), tooltip="Resets all settings on this tab to their default values.")], [sg.Column(effects_col_layout, scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]]

            advanced_pitch_settings_layout = [[sg.Text("Pitch Crossover Note"), sg.Combo(crossover_notes, key="split_pitch_crossover_display", default_value=default_display_crossover, expand_x=True, readonly=True, enable_events=True, tooltip="Sets the note where the Low and High pitch correction settings divide.\nOnly active when Split Pitch Correction is enabled.")],
                                              [sg.Frame("Low Pitch Settings", [[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="low_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("low_pitch_strength", 0.3), expand_x=True, enable_events=True)],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="low_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("low_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True)],], expand_x=True)],
                                              [sg.Frame("High Pitch Settings", [[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="high_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("high_pitch_strength", 0.3), expand_x=True, enable_events=True)],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="high_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("high_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True)],], expand_x=True)],]
            auto_pitch_settings_layout = [[sg.Text("F0 Input Smoothing"), sg.Slider(range=(0.0, 0.99), key="f0_smoothing_factor", resolution=0.01, orientation="h", default_value=data.get("f0_smoothing_factor", 0.0), expand_x=True, enable_events=True, tooltip="Smooths the raw F0 input before it is used by dynamic effects.\nHigher values can reduce artifacts from rapid pitch changes (e.g., shouting).")],
                                          [sg.HSep()],
                                          [sg.pin(sg.Column([[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="auto_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("auto_pitch_strength", 0.3), expand_x=True, enable_events=True, tooltip="How strongly the pitch is corrected towards the target.\nHigher values are more robotic.")],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="auto_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("auto_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True, tooltip="The maximum allowed pitch correction in semitones.")],], key='global_pitch_settings', visible=not use_split_pitch_correction))],
                                          [sg.Text("Pitch Stability"), sg.Slider(range=(0.0, 1.0), key="pitch_stability", resolution=0.01, orientation="h", default_value=data.get("pitch_stability", 0.2), expand_x=True, enable_events=True, tooltip="How much to smooth the pitch correction adjustments.\nHigher values are smoother but less responsive.")],
                                          [sg.Checkbox("Enable Shout Dampening", key="use_shout_dampening", default=use_shout_dampening, enable_events=True, tooltip="Lowers the target pitch when shouting to prevent an unnatural sound.")],
                                          [sg.pin(sg.Column([[sg.Text("Dampening Strength"), sg.Slider(range=(0.0, 1.0), key="shout_dampening_strength", resolution=0.01, orientation="h", default_value=data.get("shout_dampening_strength", 0.8), expand_x=True, enable_events=True)]], key='shout_dampening_slider_row', visible=use_shout_dampening))],
                                          [sg.HSep()],
                                          [sg.Checkbox("Enable Split Pitch Correction", key="use_split_pitch_correction", default=use_split_pitch_correction, enable_events=True, tooltip="Uses different correction strengths for pitches below and above the Crossover Note.")],
                                          [sg.pin(sg.Frame("Split Pitch Correction Settings", advanced_pitch_settings_layout, key="advanced_pitch_frame", expand_x=True, visible=use_split_pitch_correction))]]
            
            pitch_shaper_master_column = sg.pin(sg.Column([
                [sg.Checkbox("Enable Auto Pitch Correction", key="auto_pitch_correction", default=auto_pitch_correction_enabled, tooltip="Automatically corrects your pitch to match the selected gender profile (M->F or F->M).", enable_events=True)],
                [sg.pin(sg.Frame("Auto Pitch Settings", auto_pitch_settings_layout, key="auto_pitch_frame", expand_x=True, visible=auto_pitch_correction_enabled))]
            ], key='pitch_shaper_column'))

            auto_pitch_tab_layout = [[sg.Push(), sg.Button("Reset", key="-RESET_AUTO_PITCH-", size=(8,1), tooltip="Resets all settings on this tab to their default values.")], [pitch_shaper_master_column]]
            
            eq_band_columns = []
            eq_gains = data.get('eq_gains_db', {hz: 0.0 for hz in self.eq_freq_bands})
            for hz in self.eq_freq_bands:
                gain_val = eq_gains.get(str(hz), eq_gains.get(hz, 0.0))
                freq_label = f"{hz} Hz" if hz < 1000 else f"{hz/1000:.1f}k".replace(".0","")
                band_col = sg.Column([
                    [sg.Text(freq_label, size=(6,1), justification='center', font=("Helvetica", 9))],
                    [sg.Slider(range=(12, -12), key=f'eq_gain_{hz}hz_db', resolution=1, orientation="v", size=(4, 18), default_value=gain_val, enable_events=True, disable_number_display=True)],
                    [sg.Text(f"{gain_val:.0f} dB", key=f"eq_gain_label_{hz}hz", size=(6,1), justification='center', font=("Helvetica", 9))]
                ], element_justification='center', pad=(8, 10))
                eq_band_columns.append(band_col)

            eq_preset_list = ["Flatten", "Invert", "Bass Boost", "Treble Boost", "Bass Cut", "Treble Cut", "Studio", "HQ Phone", "LQ Phone"]
            equalizer_controls_layout = sg.Column([
                [sg.Checkbox("Enable EQ", key="enable_equalizer", default=enable_equalizer, enable_events=True, tooltip="Enables a 10-band graphic equalizer for the final output audio.")],
                [
                    sg.Text("Preset"),
                    sg.Combo(eq_preset_list, key='-EQ_PRESET_COMBO-', readonly=True, expand_x=True, tooltip="Select a preset EQ curve."),
                    sg.Button("Apply", key='-EQ_APPLY_PRESET-')
                ]
            ], pad=(10,10))

            equalizer_tab_layout = [
                [sg.Push(), sg.Button("Reset", key="-RESET_EQUALIZER-", size=(8,1), tooltip="Resets all settings on this tab to their default values.")],
                [equalizer_controls_layout],
                [sg.pin(sg.Frame("10-Band Graphic Equalizer", [eq_band_columns], key='equalizer_bands_column', visible=enable_equalizer, border_width=0, pad=(0,0), element_justification='c'))]
            ]

            performance_settings_layout = [
                [sg.Push(), sg.Button("Reset", key="-RESET_PERFORMANCE-", size=(8,1), tooltip="Resets all settings on this tab to their default values.")],
                [sg.Text("Sample Length (s)"), sg.Slider(range=(0.1, 5.0), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 1.0), expand_x=True, tooltip="The length of the audio chunk processed at one time (in seconds).\nHigher values may increase latency but improve quality.")],
                [sg.Text("Crossfade Length (ms)"), sg.Slider(range=(10, 500), key="crossfade_length", resolution=10, orientation="h", default_value=data.get("crossfade_length", 80.0), expand_x=True, tooltip="The duration of the crossfade between processed audio chunks to reduce artifacts.")],
                [sg.Text("Extra Inference Time (ms)"), sg.Slider(range=(50, 5000), key="extra_time", resolution=10, orientation="h", default_value=data.get("extra_time", 40.0), expand_x=True, tooltip="Extra audio (in ms) sent to the model to prevent the start of a word from being cut off.")],
                [sg.Text("SOLA Search (ms)"), sg.Slider(range=(2, 100), key="sola_search_ms", resolution=1, orientation="h", default_value=data.get("sola_search_ms", 10.0), expand_x=True, tooltip="Search length for the SOLA algorithm to find the best crossfade point.")],
                [sg.Text("Precision"), sg.Radio("FP16", "precision", key="fp16", default=data.get("is_half", True), tooltip="Half-precision floating point. Faster, uses less VRAM, but may be less accurate."), sg.Radio("FP32", "precision", key="fp32", default=not data.get("is_half", True), tooltip="Full-precision floating point. Slower, uses more VRAM, but is more accurate.")],
                [sg.Checkbox("Input Denoise", key="I_noise_reduce", default=data.get("I_noise_reduce", False), enable_events=True, tooltip="Apply noise reduction to the input audio."), sg.Checkbox("Output Denoise", key="O_noise_reduce", default=data.get("O_noise_reduce", False), enable_events=True, tooltip="Apply noise reduction to the final output audio.")],
                [sg.Checkbox("Enable CMD Diagnostics", key="enable_cmd_diagnostics", default=data.get("enable_cmd_diagnostics", False), enable_events=True, tooltip="Prints real-time diagnostic information to the command line.")],
                [sg.Text("Model Sample Rate:"), sg.Text("N/A", key="sg_samplerate")],
            ]

            profiles_tab_layout = [
                [sg.Text("Manage and recall your complete settings configurations.")],
                [sg.HSep()],
                [sg.Text("Profile Name", size=(12,1)), sg.Input(key='-PROFILE_NAME-', expand_x=True, tooltip="Enter a name for the profile you want to save.")],
                [sg.Button("Save Current Profile", key='-SAVE_PROFILE-', expand_x=True, tooltip="Saves all current settings as a new profile.")],
                [sg.HSep()],
                [sg.Text("Load Profile", size=(12,1)), sg.Combo([], key='-PROFILE_LIST-', expand_x=True, readonly=True, tooltip="Select a saved profile from the list.")],
                [
                    sg.Button("Load Selected", key='-LOAD_PROFILE-', tooltip="Loads the settings from the selected profile."),
                    sg.Button("Refresh List", key='-REFRESH_PROFILES-', tooltip="Rescans the folder for saved profiles."),
                    sg.Button("Delete Selected", key='-DELETE_PROFILE-', button_color=('white', '#8B0000'), tooltip="Deletes the selected profile permanently.")
                ]
            ]
            
            aboutme_tab_layout = [[sg.Column([[sg.VPush()],[sg.Text("RVC Real-time Voice Changer", font=("Helvetica", 24, "bold"), justification='center')],[sg.HorizontalSeparator()],[sg.Text("\nDiscord: @glickko", font=("Helvetica", 12), justification='center')],[sg.Text("Visit Github", text_color="#00B0F0", font=("Helvetica", 12, "underline"), enable_events=True, key='-GITHUB_LINK-', tooltip='https://github.com/Glicko-Personal/Mangio-RVC-Fork')],[sg.VPush()]], element_justification='center', expand_x=True)]]

            control_bar_layout = [
                sg.Button("Start", key="start_vc", disabled=False, size=(8,1)), 
                sg.Button("Stop", key="stop_vc", disabled=True, size=(8,1)), 
                sg.pin(sg.Frame('File Controls', [[sg.Button("Play", key="play_file", disabled=True, size=(6,1)), sg.Button("Stop", key="stop_file", disabled=True, size=(6,1)), sg.Checkbox("Loop", key="loop_audio_file", default=data.get("loop_audio_file", False), enable_events=True, tooltip="Loop the audio file when it finishes.")]], key='file_controls_frame', visible=False)),
                sg.Push(), 
                sg.Text("Avg Pitch (Hz):"), sg.Text("0.00", key="avg_hz"),
                sg.Text("Infer Time (ms):"), sg.Text("0", key="infer_time")
            ]
            
            main_column = sg.Column(main_tab_layout, scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)

            tab_group = sg.TabGroup([[
                 sg.Tab('Main', [[main_column]]), 
                 sg.Tab('Model & Devices', model_devices_tab_layout),
                 sg.Tab('Effects', effects_tab_layout, expand_x=True, expand_y=True),
                 sg.Tab('Equalizer', equalizer_tab_layout),
                 sg.Tab('Auto Pitch', auto_pitch_tab_layout), 
                 sg.Tab('Performance', performance_settings_layout),
                 sg.Tab('Profiles', profiles_tab_layout),
                 sg.Tab('About Me', aboutme_tab_layout)
            ]], expand_x=True, expand_y=True)

            layout = [[sg.Column([control_bar_layout], expand_x=True)], [sg.HSep()], [tab_group]]
            
            self.window = sg.Window("RVC Real-time Voice Changer", layout=layout, finalize=True, resizable=True)
            self._update_profile_list()
            self.update_pitch_shaper_visibility(current_profile)
            self.event_handler()

        def handle_reset(self, keys_to_reset):
            for key in keys_to_reset:
                default_value = self.default_settings.get(key)
                if default_value is None: continue

                if key == "eq_gains_db" or key == "input_eq_gains_db":
                    prefix = "input_" if key == "input_eq_gains_db" else ""
                    for hz, gain in default_value.items():
                        self.window[f'{prefix}eq_gain_{hz}hz_db'].update(gain)
                        self.window[f'{prefix}eq_gain_label_{hz}hz'].update(f"{gain:.0f} dB")
                elif isinstance(self.window.Element(key), sg.Radio):
                    if key == "default": self.window[key].update(True)
                    else: self.window[key].update(False)
                elif key == "split_pitch_crossover_display":
                    default_note = self.default_settings.get("split_pitch_crossover", "C4")
                    self.window[key].update(self.get_display_string_from_note(default_note))
                else:
                    self.window[key].update(default_value)
                
                if self.audio_processor:
                    if key in ["default", "m2f", "f2m", "f2f", "m2m"]:
                        profile_name = "Default"
                        self.audio_processor.queue_parameter_update("voice_profile", profile_name)
                        self.update_pitch_shaper_visibility(profile_name)
                    else:
                        self.audio_processor.queue_parameter_update(key, default_value)
                
                if key in self.visibility_param_keys:
                    self.update_visibility(key, default_value)
        
        def update_pitch_shaper_visibility(self, profile_name):
            is_visible = profile_name in ["Male to Female", "Female to Male"]
            self.window['pitch_shaper_column'].update(visible=is_visible)
            self.window['dynamic_timbre_master_column'].update(visible=is_visible)
            self.window['dynamic_formant_frame'].update(visible=is_visible)
            self.window['default_timbre_row'].update(visible=not is_visible)
            self.window['static_formant_row'].update(visible=not is_visible)
            
        def update_visibility(self, key, value):
            if key == "use_shout_dampening": self.window['shout_dampening_slider_row'].update(visible=value)
            elif key == "use_split_pitch_correction":
                self.window["global_pitch_settings"].update(visible=not value)
                self.window["advanced_pitch_frame"].update(visible=value)
            elif key == "use_reverb": self.window['reverb_settings_column'].update(visible=value)
            elif key == "enable_discord_effects": self.window['discord_effects_column'].update(visible=value)
            elif key == "enable_saturation_effect": self.window['saturation_column'].update(visible=value)
            elif key == "enable_cave_effect": self.window['cave_effect_column'].update(visible=value)
            elif key == "enable_low_freq_dampening": self.window['low_freq_dampening_column'].update(visible=value)
            elif key == "enable_dynamic_proximity": self.window['dynamic_proximity_column'].update(visible=value)
            elif key == "auto_pitch_correction": self.window["auto_pitch_frame"].update(visible=value)
            elif key == "enable_dynamic_timbre":
                self.window["dynamic_timbre_column"].update(visible=value)
            elif key == "enable_dynamic_formant":
                self.window["dynamic_formant_column"].update(visible=value)
            elif key == "enable_equalizer": self.window["equalizer_bands_column"].update(visible=value)
            elif key == "enable_input_equalizer": self.window["input_equalizer_column"].update(visible=value)
        
        def event_handler(self):
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    if self.audio_processor: self.audio_processor.stop()
                    self.save(values)
                    sys.exit()
                
                if isinstance(event, str) and event.startswith("-RESET_"):
                    key_map = {
                        "-RESET_MAIN-": self.main_reset_keys, "-RESET_EFFECTS-": self.effects_reset_keys,
                        "-RESET_AUTO_PITCH-": self.autopitch_reset_keys, "-RESET_PERFORMANCE-": self.performance_reset_keys,
                        "-RESET_EQUALIZER-": self.equalizer_reset_keys,
                    }
                    if event in key_map: self.handle_reset(key_map[event])
                    continue
                
                if event == '-SAVE_PROFILE-': self._save_profile(values, values['-PROFILE_NAME-'])
                elif event == '-LOAD_PROFILE-': self._load_profile(values['-PROFILE_LIST-']) if values['-PROFILE_LIST-'] else sg.popup("No profile selected.", title="Info")
                elif event == '-DELETE_PROFILE-': self._delete_profile(values['-PROFILE_LIST-']) if values['-PROFILE_LIST-'] else sg.popup("No profile selected to delete.", title="Info")
                elif event == '-REFRESH_PROFILES-': self._update_profile_list()

                elif event == "output_volume_normalization":
                    is_normalized = values["output_volume_normalization"]
                    self.window["output_volume"].update(disabled=is_normalized)
                    new_volume = 0.0 if is_normalized else values["output_volume"]
                    if is_normalized: self.window["output_volume"].update(value=new_volume)
                    if self.audio_processor:
                        self.audio_processor.queue_parameter_update("output_volume", new_volume)
                        self.audio_processor.queue_parameter_update("output_volume_normalization", is_normalized)

                elif event in self.realtime_param_keys:
                    if self.audio_processor:
                        param_key = event
                        new_value = values[event]
                        if event == "split_pitch_crossover_display":
                            param_key = "split_pitch_crossover"
                            new_value = self.get_note_from_display_string(values[event])
                        self.audio_processor.queue_parameter_update(param_key, new_value)
                    if event in self.visibility_param_keys: self.update_visibility(event, values[event])

                elif event in ["default", "m2f", "f2m", "f2f", "m2m"]:
                    profile = "Default"
                    if values['m2f']: profile = "Male to Female"
                    elif values['f2m']: profile = "Female to Male"
                    elif values['f2f']: profile = "Female to Female"
                    elif values['m2m']: profile = "Male to Male"
                    self.update_pitch_shaper_visibility(profile)
                    if self.audio_processor:
                        self.audio_processor.queue_parameter_update("voice_profile", profile)

                elif event == '-UPDATE_STATUS-':
                    total_time_ms, avg_hz = values[event]
                    self.window["infer_time"].update(f"{int(total_time_ms)}"); self.window["avg_hz"].update(f"{avg_hz:.2f}")
                
                elif event == "refresh_devices":
                    input_devices, output_devices, _, _ = self.get_devices(update=True)
                    self.window["sg_input_device"].update(values=input_devices, value=values['sg_input_device'])
                    self.window["sg_output_device"].update(values=output_devices, value=values['sg_output_device'])
                    sg.popup("Device list has been refreshed.")
                elif event in ("mic_input", "file_input"):
                    self.window['audio_file_row'].update(visible=values["file_input"]); self.window['input_device_col'].update(visible=values["mic_input"])
                elif event == "use_index_file":
                    self.window['index_path_row'].update(visible=values["use_index_file"]); self.window['index_rate_row'].update(visible=values["use_index_file"])
                elif event == "start_vc":
                    if self.set_values(values):
                        self.start_vc()
                        if self.config.input_source == 'file' and values['input_audio_path']: self.audio_processor.request_load_file(self.config.input_audio_path)
                elif event == "stop_vc": self.save(values); self.stop_vc()
                elif event == "play_file":
                    if self.audio_processor: self.audio_processor.play_file()
                elif event == "stop_file":
                    if self.audio_processor: self.audio_processor.stop_file()
                elif event == "loop_audio_file":
                    if self.audio_processor: 
                        self.audio_processor.queue_parameter_update(event, values[event])
                elif event == '-GITHUB_LINK-':
                    webbrowser.open('https://github.com/Glicko-Personal/Mangio-RVC-Fork')


            self.window.close()

        def stop_vc(self):
            if self.audio_processor:
                self.audio_processor.stop(); self.audio_processor = None
            self.window["sg_samplerate"].update("N/A"); self.window["start_vc"].update(disabled=False)
            self.window["stop_vc"].update(disabled=True); self.window["infer_time"].update("0")
            self.window["avg_hz"].update("0.00"); self.window['file_controls_frame'].update(visible=False)
        
        def set_values(self, values):
            if not values["pth_path"].strip(): sg.popup("Please select a .pth file"); return False
            self.config.use_index_file = values["use_index_file"]
            if self.config.use_index_file and (not values["index_path"].strip()): sg.popup("Please select an .index file or uncheck 'Use Index File'"); return False
            if re.search("[\u4e00-\u9fa5]+", values["pth_path"]) or (self.config.use_index_file and re.search("[\u4e00-\u9fa5]+", values["index_path"])): sg.popup("File paths cannot contain non-ASCII characters"); return False
            
            self.config.input_source = "file" if values['file_input'] else "microphone"
            self.config.input_audio_path = values["input_audio_path"]
            self.set_devices(values["sg_input_device"], values["sg_output_device"])

            for key in self.realtime_param_keys:
                if key == "split_pitch_crossover_display":
                    self.config.split_pitch_crossover = self.get_note_from_display_string(values[key])
                elif key.startswith('eq_gain_') or key.startswith('input_eq_gain_'):
                    continue
                elif key in values: 
                    setattr(self.config, key, values[key])
            
            self.config.eq_gains_db = {hz: values.get(f'eq_gain_{hz}hz_db', 0.0) for hz in self.eq_freq_bands}
            self.config.input_eq_gains_db = {hz: values.get(f'input_eq_gain_{hz}hz_db', 0.0) for hz in self.eq_freq_bands}

            profile = "Default"
            if values['m2f']: profile = "Male to Female"
            elif values['f2m']: profile = "Female to Male"
            self.config.voice_profile = profile

            self.config.pth_path, self.config.index_path = values["pth_path"], values["index_path"]
            self.config.is_half = values["fp16"]
            self.config.device_samplerate, self.config.sola_search_ms = int(values["device_samplerate"]), values["sola_search_ms"] / 1000
            self.config.block_time, self.config.crossfade_time, self.config.extra_time = values["block_time"], values["crossfade_length"] / 1000, values["extra_time"] / 1000
            return True

        def start_vc(self):
            torch.cuda.empty_cache(); print("---STARTING RVC---")
            try:
                rvc = RVC(self.config)
                print("---RVC CLASS INITIALIZED SUCCESSFULLY---")
                self.config.samplerate = rvc.tgt_sr
                self.window["sg_samplerate"].update(f"{rvc.tgt_sr} Hz")
                self.audio_processor = AudioProcessor(self.config, rvc, self.window)
                self.audio_processor.start()
                self.window["start_vc"].update(disabled=True); self.window["stop_vc"].update(disabled=False)
            except Exception as e:
                print(f"!!! FAILED TO INITIALIZE RVC CLASS !!!\n{traceback.format_exc()}"); sg.popup(f"Failed to initialize RVC: {e}")

        def get_devices(self, update: bool = True):
            if update: sd._terminate(); sd._initialize()
            devices, hostapis = sd.query_devices(), sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]: devices[device_idx]["hostapi_name"] = hostapi["name"]
            input_devices = [f"{d['name']} ({d['hostapi_name']})" for d in devices if d["max_input_channels"] > 0]
            output_devices = [f"{d['name']} ({d['hostapi_name']})" for d in devices if d["max_output_channels"] > 0]
            input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
            output_devices_indices = [d["index"] for d in devices if d["max_output_channels"] > 0]
            return (input_devices, output_devices, input_devices_indices, output_devices_indices)

        def set_devices(self, input_device, output_device):
            try:
                if input_device:
                    input_devices, _, input_device_indices, _ = self.get_devices(update=False)
                    sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
                if output_device:
                    _, output_devices, _, output_device_indices = self.get_devices(update=False)
                    sd.default.device[1] = output_device_indices[output_devices.index(output_device)]
            except Exception as e:
                sg.popup(f"Error setting devices: {e}")

    gui = GUI()