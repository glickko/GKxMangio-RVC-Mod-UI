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
            self.output_volume: float = 0.0
            self.auto_pitch_correction: bool = False
            self.pitch_stability: float = 0.2
            self.auto_pitch_strength: float = 0.3
            self.auto_pitch_max_adjustment: float = 2.0
            self.voice_profile: str = "Default"
            self.use_shout_dampening: bool = False
            self.shout_dampening_strength: float = 0.8
            self.formant_shift: float = 1.0
            self.timbre: float = 1.0
            self.use_auto_timbre: bool = False
            self.auto_timbre_shout_hz: float = 300.0
            self.auto_timbre_strength: float = 0.7
            self.use_split_pitch_correction: bool = False
            self.split_pitch_crossover: str = "C4"
            self.low_pitch_strength: float = 0.3
            self.low_pitch_max_adjustment: float = 2.0
            self.high_pitch_strength: float = 0.3
            self.high_pitch_max_adjustment: float = 2.0
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


    class GUI:
        def __init__(self) -> None:
            self.config = GUIConfig()
            self.audio_processor: AudioProcessor = None
            self.crossover_notes_list = self.get_crossover_notes()
            self.realtime_param_keys = []
            self.default_settings = {}
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
                "input_volume": 1.0, "output_volume": 0.0,
                "auto_pitch_correction": False, "pitch_stability": 0.2, "auto_pitch_strength": 0.3,
                "auto_pitch_max_adjustment": 2.0, "voice_profile": "Default",
                "use_shout_dampening": False, "shout_dampening_strength": 0.8,
                "formant_shift": 1.0, "timbre": 1.0, "use_auto_timbre": False,
                "auto_timbre_shout_hz": 300.0, "auto_timbre_strength": 0.7,
                "use_split_pitch_correction": False, "split_pitch_crossover": "C4",
                "low_pitch_strength": 0.3, "low_pitch_max_adjustment": 2.0,
                "high_pitch_strength": 0.3, "high_pitch_max_adjustment": 2.0,
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
                "input_source": "microphone", "input_audio_path": "", "loop_audio_file": False
            }

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices(update=True)
            try:
                with open("values1.json", "r") as j:
                    data = json.load(j)
                    default_data = self.get_default_settings(input_devices, output_devices)
                    for key, value in default_data.items(): data.setdefault(key, value)
            except:
                data = self.get_default_settings(input_devices, output_devices)
                with open("values1.json", "w") as j:
                    json.dump(data, j, indent=4)
            return data
        
        def save(self, values):
            if not values:
                return
            profile = "Default"
            if values.get('m2f'): profile = "Male to Female"
            elif values.get('f2m'): profile = "Female to Male"
            elif values.get('f2f'): profile = "Female to Female"
            elif values.get('m2m'): profile = "Male to Male"
            values['voice_profile'] = profile
            
            values['input_source'] = "file" if values.get('file_input') else "microphone"
            values["split_pitch_crossover"] = self.get_note_from_display_string(values.get("split_pitch_crossover_display", "C4"))
            
            temp_values = {k: v for k, v in values.items() if k != "split_pitch_crossover_display"}
            with open("values1.json", "w") as j:
                json.dump(temp_values, j, indent=4)
            print("Settings saved to values1.json")

        def launcher(self):
            data = self.load()
            sg.theme("DarkGrey9")
            input_devices, output_devices, _, _ = self.get_devices(update=False)
            self.default_settings = self.get_default_settings(input_devices, output_devices)
            samplerate_options = [16000, 32000, 40000, 44100, 48000, 96000]
            
            crossover_notes = self.crossover_notes_list
            default_display_crossover = self.get_display_string_from_note(data.get("split_pitch_crossover", "C4"))

            use_index_file = data.get("use_index_file", True)
            current_profile = data.get("voice_profile", "Default")
            use_shout_dampening = data.get("use_shout_dampening", False)
            auto_pitch_correction_enabled = data.get("auto_pitch_correction", False)
            use_auto_timbre = data.get("use_auto_timbre", False)
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
                "threhold", "pitch", "index_rate", "input_volume", "output_volume", "I_noise_reduce", "O_noise_reduce", "formant_shift", "timbre",
                "auto_pitch_correction", "pitch_stability", "auto_pitch_strength", "auto_pitch_max_adjustment", "use_shout_dampening", "shout_dampening_strength",
                "use_auto_timbre", "auto_timbre_shout_hz", "auto_timbre_strength",
                "use_split_pitch_correction", "low_pitch_strength", "low_pitch_max_adjustment", "high_pitch_strength", "high_pitch_max_adjustment", "split_pitch_crossover_display",
                "use_reverb", "reverb_room_size", "reverb_damping", "reverb_wet_level", "reverb_dry_level",
                "enable_discord_effects", "discord_proximity", "discord_noise", "discord_quality",
                "enable_phone_effect", "enable_saturation_effect", "saturation_threshold_hz", "saturation_drive_db",
                "enable_cave_effect", "cave_delay_time", "cave_feedback", "cave_mix",
                "enable_low_freq_dampening", "low_freq_dampening_threshold_hz", "low_freq_dampening_level_db",
                "enable_dynamic_proximity", "dynamic_proximity_strength", "dynamic_proximity_room_size", "enable_cmd_diagnostics"
            ]
            self.visibility_param_keys = ["use_shout_dampening", "use_split_pitch_correction", "use_reverb", "enable_discord_effects", "enable_saturation_effect", "enable_cave_effect", "enable_low_freq_dampening", "enable_dynamic_proximity", "auto_pitch_correction", "enable_phone_effect", "use_auto_timbre"]

            self.realtime_param_keys = [
                "threhold", "pitch", "index_rate", "input_volume", "output_volume", "I_noise_reduce", "O_noise_reduce", "formant_shift", "timbre",
                "auto_pitch_correction", "pitch_stability", "auto_pitch_strength", "auto_pitch_max_adjustment", "use_shout_dampening", "shout_dampening_strength",
                "use_auto_timbre", "auto_timbre_shout_hz", "auto_timbre_strength", "auto_timbre_stability",
                "use_split_pitch_correction", "low_pitch_strength", "low_pitch_max_adjustment", "high_pitch_strength", "high_pitch_max_adjustment", "split_pitch_crossover_display",
                "use_reverb", "reverb_room_size", "reverb_damping", "reverb_wet_level", "reverb_dry_level",
                "enable_discord_effects", "discord_proximity", "discord_noise", "discord_quality",
                "enable_phone_effect", "enable_saturation_effect", "saturation_threshold_hz", "saturation_drive_db",
                "enable_cave_effect", "cave_delay_time", "cave_feedback", "cave_mix",
                "enable_low_freq_dampening", "low_freq_dampening_threshold_hz", "low_freq_dampening_level_db",
                "enable_dynamic_proximity", "dynamic_proximity_strength", "dynamic_proximity_room_size", "enable_cmd_diagnostics"
            ]
            self.visibility_param_keys = ["use_shout_dampening", "use_split_pitch_correction", "use_reverb", "enable_discord_effects", "enable_saturation_effect", "enable_cave_effect", "enable_low_freq_dampening", "enable_dynamic_proximity", "auto_pitch_correction", "enable_phone_effect", "use_auto_timbre"]

            self.main_reset_keys = ["threhold", "pitch", "index_rate", "input_volume", "output_volume", "auto_pitch_correction", "use_auto_timbre", "default", "m2f", "f2m", "f2f", "m2m"]
            self.effects_reset_keys = ["formant_shift", "timbre", "auto_timbre_shout_hz", "auto_timbre_strength", "auto_timbre_stability", "enable_low_freq_dampening", "low_freq_dampening_threshold_hz", "low_freq_dampening_level_db", "enable_saturation_effect", "saturation_threshold_hz", "saturation_drive_db", "enable_dynamic_proximity", "dynamic_proximity_strength", "dynamic_proximity_room_size", "enable_discord_effects", "discord_proximity", "discord_noise", "discord_quality", "enable_cave_effect", "cave_delay_time", "cave_feedback", "cave_mix", "use_reverb", "reverb_room_size", "reverb_damping", "reverb_wet_level", "reverb_dry_level", "enable_phone_effect"]
            self.autopitch_reset_keys = ["auto_pitch_strength", "auto_pitch_max_adjustment", "pitch_stability", "use_shout_dampening", "shout_dampening_strength", "use_split_pitch_correction", "split_pitch_crossover_display", "low_pitch_strength", "low_pitch_max_adjustment", "high_pitch_strength", "high_pitch_max_adjustment"]
            self.performance_reset_keys = ["I_noise_reduce", "O_noise_reduce", "enable_cmd_diagnostics"]

            profile_checkboxes_layout = [
                sg.pin(sg.Column([[ 
                    sg.Checkbox("Auto Pitch Correction", key="auto_pitch_correction", default=auto_pitch_correction_enabled, tooltip="Auto-adjust pitch to profile.", enable_events=True),
                    sg.Checkbox("Auto Timbre", key="use_auto_timbre", default=use_auto_timbre, tooltip="Automatically adjust timbre on high pitch to reduce 'chipmunk' effect.", enable_events=True)
                ]], key="profile_checkboxes_column", visible=current_profile not in ["Default", "Female to Female", "Male to Male"]))
            ]

            main_tab_layout = [
                [sg.Push(), sg.Button("Reset", key="-RESET_MAIN-", size=(8,1))],
                [sg.Text("Response Threshold"), sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -30), expand_x=True, tooltip="Noise gate.", enable_events=True)],
                [sg.Text("Pitch Setting"), sg.Slider(range=(-24, 24), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), expand_x=True, tooltip="Global pitch shift.", enable_events=True)],
                [sg.pin(sg.Column([[sg.Text("Index Rate"), sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0.3), expand_x=True, tooltip="Timbre mixing amount.", enable_events=True) ]], key='index_rate_row', visible=use_index_file))],
                [sg.Text("Input Volume"), sg.Slider(range=(0.0, 2.0), key="input_volume", resolution=0.01, orientation="h", default_value=data.get("input_volume", 1.0), expand_x=True, tooltip="Microphone volume.", enable_events=True)],
                [sg.Text("Output Volume (dB)"), sg.Slider(range=(-60, 12), key="output_volume", resolution=1, orientation="h", default_value=data.get("output_volume", 0.0), expand_x=True, tooltip="Final output volume.", enable_events=True)],
                [sg.Frame("Voice Profile", [[ 
                    sg.Radio("Default", "voice_profile", key="default", default=current_profile == "Default", enable_events=True),
                    sg.Radio("M->F", "voice_profile", key="m2f", default=current_profile == "Male to Female", enable_events=True), 
                    sg.Radio("F->M", "voice_profile", key="f2m", default=current_profile == "Female to Male", enable_events=True), 
                    sg.Radio("F->F", "voice_profile", key="f2f", default=current_profile == "Female to Female", enable_events=True), 
                    sg.Radio("M->M", "voice_profile", key="m2m", default=current_profile == "Male to Male", enable_events=True)
                ]], expand_x=True)],
                profile_checkboxes_layout,
                [sg.Text("Pitch Algorithm: rmvpe (recommended)")]
            ]

            model_layout = [
                [sg.Checkbox("Use Index File", key="use_index_file", default=use_index_file, enable_events=True, tooltip="Use .index file for timbre.")],
                [sg.Input(default_text=data.get("pth_path", ""), key="pth_path", expand_x=True), sg.FileBrowse("Select .pth file", initial_folder=os.path.join(os.getcwd(), "weights"), file_types=(("PTH files", "*.pth"),))],
                [sg.pin(sg.Column([[sg.Input(default_text=data.get("index_path", ""), key="index_path", expand_x=True), sg.FileBrowse("Select .index file", initial_folder=os.path.join(os.getcwd(), "logs"), file_types=(("Index files", "*.index"),)) ]], key='index_path_row', visible=use_index_file))]
            ]
            
            input_source_layout = [
                [sg.Radio("Microphone", "input_source", key="mic_input", default=current_input_source == "microphone", enable_events=True),
                 sg.Radio("Audio File", "input_source", key="file_input", default=current_input_source == "file", enable_events=True)],
                [sg.pin(sg.Column([[ 
                    sg.Input(default_text=data.get("input_audio_path", ""), key="input_audio_path", expand_x=True, enable_events=True), 
                    sg.FileBrowse("Select Audio File", file_types=(("Audio Files", "*.wav *.mp3 *.flac"),))
                ]], key='audio_file_row', visible=current_input_source == "file"))]
            ]
            
            devices_layout = [
                [sg.pin(sg.Column([[ 
                    sg.Text("Input Device"), sg.Combo(input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), expand_x=True, tooltip="Select your microphone.")
                ]], key='input_device_col', visible=current_input_source == "microphone"))],
                [sg.Text("Output Device"), sg.Combo(output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), expand_x=True, tooltip="Select your speakers.", enable_events=True)],
                [sg.Text("Device Sample Rate"), sg.Combo(samplerate_options, key="device_samplerate", default_value=data.get("device_samplerate", 48000), expand_x=True), sg.Button("Refresh", key="refresh_devices")]
            ]
            
            model_devices_tab_layout = [[sg.Frame("Load Model", model_layout, expand_x=True)], [sg.Frame("Input Source", input_source_layout, expand_x=True)], [sg.Frame("Audio Devices", devices_layout, expand_x=True)]]

            formant_timbre_frame = sg.Frame("Formant & Timbre", [[sg.Text("Formant Shift"), sg.Slider(range=(0.0, 2.0), key="formant_shift", resolution=0.01, orientation="h", default_value=data.get("formant_shift", 1.0), expand_x=True, enable_events=True)], [sg.Text("Timbre"), sg.Slider(range=(0.0, 2.0), key="timbre", resolution=0.01, orientation="h", default_value=data.get("timbre", 1.0), expand_x=True, enable_events=True)],], expand_x=True)
            auto_timbre_frame = sg.pin(sg.Frame("Auto Timbre Settings", [[sg.Text("Shout Threshold (Hz)"), sg.Slider(range=(150, 800), key="auto_timbre_shout_hz", resolution=10, orientation="h", default_value=data.get("auto_timbre_shout_hz", 300.0), expand_x=True, enable_events=True)], [sg.Text("Adjustment Strength"), sg.Slider(range=(0.0, 10.0), key="auto_timbre_strength", resolution=0.01, orientation="h", default_value=data.get("auto_timbre_strength", 0.7), expand_x=True, enable_events=True, tooltip="Controls how much the timbre is lowered on high pitches. Higher value = stronger reduction. At 10, one octave of over-pitch will result in a ~5 semitone formant shift.")]], key='auto_timbre_settings_frame', expand_x=True, visible=use_auto_timbre and current_profile not in ["Default", "Female to Female", "Male to Male"]))
            low_freq_dampening_frame = sg.Frame("Low Freq Dampening", [[sg.Checkbox("Enable", key="enable_low_freq_dampening", default=enable_low_freq_dampening, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Threshold (Hz)"), sg.Slider(range=(50, 200), key="low_freq_dampening_threshold_hz", resolution=1, orientation="h", default_value=data.get("low_freq_dampening_threshold_hz", 100.0), expand_x=True, enable_events=True)], [sg.Text("Dampen (dB)"), sg.Slider(range=(-24.0, 0.0), key="low_freq_dampening_level_db", resolution=0.5, orientation="h", default_value=data.get("low_freq_dampening_level_db", -6.0), expand_x=True, enable_events=True)],], key='low_freq_dampening_column', visible=enable_low_freq_dampening))]], expand_x=True)
            saturation_frame = sg.Frame("Saturation", [[sg.Checkbox("Enable", key="enable_saturation_effect", default=enable_saturation_effect, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Threshold (Hz)"), sg.Slider(range=(100, 2000), key="saturation_threshold_hz", resolution=10, orientation="h", default_value=data.get("saturation_threshold_hz", 800.0), expand_x=True, enable_events=True)], [sg.Text("Drive (dB)"), sg.Slider(range=(0.0, 24.0), key="saturation_drive_db", resolution=0.5, orientation="h", default_value=data.get("saturation_drive_db", 6.0), expand_x=True, enable_events=True)],], key='saturation_column', visible=enable_saturation_effect))]], expand_x=True)
            dynamic_proximity_frame = sg.Frame("Dynamic Smartphone Proximity", [[sg.Checkbox("Enable", key="enable_dynamic_proximity", default=enable_dynamic_proximity, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Strength"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_strength", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_strength", 0.5), expand_x=True, enable_events=True)], [sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_room_size", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_room_size", 0.2), expand_x=True, enable_events=True)],], key='dynamic_proximity_column', visible=enable_dynamic_proximity))]], expand_x=True)
            discord_effects_frame = sg.Frame("Discord Effects", [[sg.Checkbox("Enable", key="enable_discord_effects", default=enable_discord_effects, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Proximity"), sg.Slider(range=(0.0, 1.0), key="discord_proximity", resolution=0.01, orientation="h", default_value=data.get("discord_proximity", 1.0), expand_x=True, enable_events=True)], [sg.Text("Noise (dB)"), sg.Slider(range=(-80.0, 12.0), key="discord_noise", resolution=0.5, orientation="h", default_value=data.get("discord_noise", -80.0), expand_x=True, enable_events=True)], [sg.Text("Quality"), sg.Slider(range=(0.0, 1.0), key="discord_quality", resolution=0.01, orientation="h", default_value=data.get("discord_quality", 1.0), expand_x=True, enable_events=True)],], key='discord_effects_column', visible=enable_discord_effects))]], expand_x=True)
            cave_effect_frame = sg.Frame("Cave/Large Room Echo", [[sg.Checkbox("Enable", key="enable_cave_effect", default=enable_cave_effect, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Delay (ms)"), sg.Slider(range=(50, 1000), key="cave_delay_time", resolution=10, orientation="h", default_value=data.get("cave_delay_time", 250.0), expand_x=True, enable_events=True)], [sg.Text("Feedback"), sg.Slider(range=(0.0, 0.9), key="cave_feedback", resolution=0.01, orientation="h", default_value=data.get("cave_feedback", 0.4), expand_x=True, enable_events=True)], [sg.Text("Mix"), sg.Slider(range=(0.0, 1.0), key="cave_mix", resolution=0.01, orientation="h", default_value=data.get("cave_mix", 0.5), expand_x=True, enable_events=True)],], key='cave_effect_column', visible=enable_cave_effect))]], expand_x=True)
            reverb_frame = sg.Frame("Reverb", [[sg.Checkbox("Enable", key="use_reverb", default=use_reverb, enable_events=True)], [sg.pin(sg.Column([[sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="reverb_room_size", resolution=0.01, orientation="h", default_value=data.get("reverb_room_size", 0.5), expand_x=True, enable_events=True)], [sg.Text("Damping"), sg.Slider(range=(0.0, 1.0), key="reverb_damping", resolution=0.01, orientation="h", default_value=data.get("reverb_damping", 0.5), expand_x=True, enable_events=True)], [sg.Text("Wet Level"), sg.Slider(range=(0.0, 1.0), key="reverb_wet_level", resolution=0.01, orientation="h", default_value=data.get("reverb_wet_level", 0.33), expand_x=True, enable_events=True)], [sg.Text("Dry Level"), sg.Slider(range=(0.0, 1.0), key="reverb_dry_level", resolution=0.01, orientation="h", default_value=data.get("reverb_dry_level", 0.4), expand_x=True, enable_events=True)],], key='reverb_settings_column', visible=use_reverb))]], expand_x=True)
            phone_effect_frame = sg.Frame("Phone Effect", [[sg.Checkbox("Enable", key="enable_phone_effect", default=enable_phone_effect, enable_events=True)]], expand_x=True)
            effects_col_layout = [[formant_timbre_frame], [auto_timbre_frame], [dynamic_proximity_frame], [low_freq_dampening_frame], [saturation_frame], [discord_effects_frame], [cave_effect_frame], [reverb_frame], [phone_effect_frame], [sg.Frame('', [[]], size=(10, 800), border_width=0, pad=(0,0))]]
            effects_tab_layout = [[sg.Push(), sg.Button("Reset", key="-RESET_EFFECTS-", size=(8,1))], [sg.Column(effects_col_layout, scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]]

            advanced_pitch_settings_layout = [[sg.Text("Pitch Crossover Note"), sg.Combo(crossover_notes, key="split_pitch_crossover_display", default_value=default_display_crossover, expand_x=True, readonly=True, enable_events=True)],[sg.Frame("Low Pitch Settings", [[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="low_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("low_pitch_strength", 0.3), expand_x=True, enable_events=True)],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="low_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("low_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True)],], expand_x=True)],[sg.Frame("High Pitch Settings", [[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="high_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("high_pitch_strength", 0.3), expand_x=True, enable_events=True)],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="high_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("high_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True)],], expand_x=True)],]
            auto_pitch_settings_layout = [[sg.pin(sg.Column([[sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="auto_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("auto_pitch_strength", 0.3), expand_x=True, enable_events=True)],[sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 36.0), key="auto_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("auto_pitch_max_adjustment", 2.0), expand_x=True, enable_events=True)],], key='global_pitch_settings', visible=not use_split_pitch_correction))],[sg.Text("Pitch Stability"), sg.Slider(range=(0.0, 1.0), key="pitch_stability", resolution=0.01, orientation="h", default_value=data.get("pitch_stability", 0.2), expand_x=True, enable_events=True)],[sg.Checkbox("Enable Shout Dampening", key="use_shout_dampening", default=use_shout_dampening, enable_events=True)],[sg.pin(sg.Column([[sg.Text("Shout Dampening Strength"), sg.Slider(range=(0.0, 1.0), key="shout_dampening_strength", resolution=0.01, orientation="h", default_value=data.get("shout_dampening_strength", 0.8), expand_x=True, enable_events=True)]], key='shout_dampening_slider_row', visible=use_shout_dampening))],[sg.HorizontalSeparator()],[sg.Checkbox("Enable Split Pitch Correction (Advanced)", key="use_split_pitch_correction", default=use_split_pitch_correction, enable_events=True)],[sg.pin(sg.Frame("Split Pitch Correction Settings", advanced_pitch_settings_layout, key="advanced_pitch_frame", expand_x=True, visible=use_split_pitch_correction))]]
            auto_pitch_tab_layout = [[sg.Push(), sg.Button("Reset", key="-RESET_AUTO_PITCH-", size=(8,1))], [sg.pin(sg.Frame("Auto Pitch Settings", auto_pitch_settings_layout, key="auto_pitch_frame", expand_x=True, visible=auto_pitch_correction_enabled and current_profile in ["Male to Female", "Female to Male"]))]]
            
            performance_settings_layout = [
                [sg.Push(), sg.Button("Reset", key="-RESET_PERFORMANCE-", size=(8,1))],
                [sg.Text("Sample Length (s)"), sg.Slider(range=(0.1, 5.0), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 1.0), expand_x=True)],
                
                [sg.Text("Crossfade Length (ms)"), sg.Slider(range=(10, 500), key="crossfade_length", resolution=10, orientation="h", default_value=data.get("crossfade_length", 80.0), expand_x=True)],
                [sg.Text("Extra Inference Time (ms)"), sg.Slider(range=(50, 5000), key="extra_time", resolution=10, orientation="h", default_value=data.get("extra_time", 40.0), expand_x=True)],
                [sg.Text("SOLA Search (ms)"), sg.Slider(range=(2, 100), key="sola_search_ms", resolution=1, orientation="h", default_value=data.get("sola_search_ms", 10.0), expand_x=True)],
                [sg.Text("Precision"), sg.Radio("FP16", "precision", key="fp16", default=data.get("is_half", True)), sg.Radio("FP32", "precision", key="fp32", default=not data.get("is_half", True))],
                [sg.Checkbox("Input Denoise", key="I_noise_reduce", default=data.get("I_noise_reduce", False), enable_events=True), sg.Checkbox("Output Denoise", key="O_noise_reduce", default=data.get("O_noise_reduce", False), enable_events=True)],
                [sg.Checkbox("Enable CMD Diagnostics", key="enable_cmd_diagnostics", default=data.get("enable_cmd_diagnostics", False), enable_events=True)],
                [sg.Text("Model Sample Rate:"), sg.Text("N/A", key="sg_samplerate")],
            ]

            aboutme_tab_layout = [[sg.Column([[sg.VPush()],[sg.Text("RVC Real-time Voice Changer", font=("Helvetica", 24, "bold"), justification='center')],[sg.HorizontalSeparator()],[sg.Text("\nDiscord: @glickko", font=("Helvetica", 12), justification='center')],[sg.Text("Visit Github", text_color="#00B0F0", font=("Helvetica", 12, "underline"), enable_events=True, key='-GITHUB_LINK-', tooltip='https://github.com/Glicko-Personal/Mangio-RVC-Fork')],[sg.VPush()]], element_justification='center', expand_x=True)]]

            control_bar_layout = [
                sg.Button("Start", key="start_vc", disabled=False, size=(8,1)), 
                sg.Button("Stop", key="stop_vc", disabled=True, size=(8,1)), 
                sg.pin(sg.Frame('File Controls', [[sg.Button("Play", key="play_file", disabled=True, size=(6,1)), sg.Button("Stop", key="stop_file", disabled=True, size=(6,1)), sg.Checkbox("Loop", key="loop_audio_file", default=data.get("loop_audio_file", False), enable_events=True)]], key='file_controls_frame', visible=False)),
                sg.Push(), 
                sg.Text("Avg Pitch (Hz):"), sg.Text("0.00", key="avg_hz"),
                sg.Text("Infer Time (ms):"), sg.Text("0", key="infer_time")
            ]
            
            tab_group = sg.TabGroup([[ 
                 sg.Tab('Main', main_tab_layout), sg.Tab('Model & Devices', model_devices_tab_layout),
                 sg.Tab('Effects', effects_tab_layout, expand_x=True, expand_y=True),
                 sg.Tab('Auto Pitch', auto_pitch_tab_layout), sg.Tab('Performance', performance_settings_layout),
                 sg.Tab('About Me', aboutme_tab_layout)
            ]], expand_x=True, expand_y=True)

            layout = [[sg.Column([control_bar_layout], expand_x=True)], [sg.HSep()], [tab_group]]
            
            self.window = sg.Window("RVC Real-time Voice Changer", layout=layout, finalize=True, resizable=True)
            self.event_handler()

        def handle_reset(self, keys_to_reset):
            for key in keys_to_reset:
                default_value = self.default_settings.get(key)
                if default_value is None: continue

                if isinstance(self.window[key], sg.Radio):
                    if key == "default": self.window[key].update(True)
                    else: self.window[key].update(False)
                elif key == "split_pitch_crossover_display":
                    default_note = self.default_settings.get("split_pitch_crossover", "C4")
                    self.window[key].update(self.get_display_string_from_note(default_note))
                else:
                    self.window[key].update(default_value)
                
                if self.audio_processor:
                    if key in ["default", "m2f", "f2m", "f2f", "m2m"]:
                        self.audio_processor.queue_parameter_update("voice_profile", self.default_settings["voice_profile"])
                    else:
                        self.audio_processor.queue_parameter_update(key, default_value)
                
                if key in self.visibility_param_keys:
                    self.update_visibility(key, default_value)
        
        def update_visibility(self, key, value):
            is_auto_pitch_profile = self.window["m2f"].get() or self.window["f2m"].get()
            is_any_profile = not self.window["default"].get()

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
            elif key == "auto_pitch_correction": self.window["auto_pitch_frame"].update(visible=value and is_auto_pitch_profile)
            elif key == "use_auto_timbre": self.window["auto_timbre_settings_frame"].update(visible=value and is_any_profile)

        def event_handler(self):
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    if self.audio_processor: self.audio_processor.stop()
                    self.save(values)
                    sys.exit()
                
                if event.startswith("-RESET_"):
                    key_map = {
                        "-RESET_MAIN-": self.main_reset_keys, "-RESET_EFFECTS-": self.effects_reset_keys,
                        "-RESET_AUTO_PITCH-": self.autopitch_reset_keys, "-RESET_PERFORMANCE-": self.performance_reset_keys
                    }
                    if event in key_map: self.handle_reset(key_map[event])
                    continue

                if event in self.realtime_param_keys:
                    if self.audio_processor:
                        param_key = event
                        new_value = values[event]
                        if event == "split_pitch_crossover_display":
                            param_key = "split_pitch_crossover"
                            new_value = self.get_note_from_display_string(values[event])
                        
                        self.audio_processor.queue_parameter_update(param_key, new_value)
                        print(f"[PARAM CHANGE] '{param_key}' updated to '{new_value}'")

                    if event in self.visibility_param_keys: self.update_visibility(event, values[event])
                    continue

                if event in ["default", "m2f", "f2m", "f2f", "m2m"]:
                    is_auto_pitch_profile = values["m2f"] or values["f2m"]
                    is_any_profile = not values["default"]
                    
                    self.window["profile_checkboxes_column"].update(visible=is_any_profile)
                    self.update_visibility("auto_pitch_correction", values["auto_pitch_correction"])
                    self.update_visibility("use_auto_timbre", values["use_auto_timbre"])

                    if self.audio_processor:
                        profile = "Default"
                        if values['m2f']: profile = "Male to Female"
                        elif values['f2m']: profile = "Female to Male"
                        elif values['f2f']: profile = "Female to Female"
                        elif values['m2m']: profile = "Male to Male"
                        self.audio_processor.queue_parameter_update("voice_profile", profile)
                        print(f"[PARAM CHANGE] 'voice_profile' updated to '{profile}'")
                    continue

                if event.startswith('-') and event.endswith('-'):
                    if event == '-UPDATE_STATUS-':
                        total_time_ms, avg_hz = values[event]
                        self.window["infer_time"].update(f"{int(total_time_ms)}"); self.window["avg_hz"].update(f"{avg_hz:.2f}")
                    elif event == '-FILE_PLAYBACK_STARTED-':
                        self.window["play_file"].update(disabled=True); self.window["stop_file"].update(disabled=False)
                    elif event == '-FILE_PLAYBACK_STOPPED-':
                        self.window["play_file"].update(text="Play", disabled=False); self.window["stop_file"].update(disabled=True)
                    elif event == '-AUDIO_FILE_LOADING-':
                        self.window['play_file'].update(text="Loading...", disabled=True); self.window['stop_file'].update(disabled=True)
                    elif event == '-AUDIO_FILE_LOADED-':
                        success, message = values[event]
                        if success: self.window['play_file'].update(text="Play", disabled=False)
                        else: self.window['play_file'].update(text="Load Failed", disabled=True); sg.popup(f"Failed to load audio file: {message}", title="Error")
                    elif event == '-STREAM_ERROR-':
                        sg.popup(f"Audio stream error: {values[event]}\nPlease check audio devices and restart.", title="Error"); self.stop_vc()
                    elif event == '-GITHUB_LINK-':
                        webbrowser.open('https://github.com/Glicko-Personal/Mangio-RVC-Fork')
                    continue

                if event == "refresh_devices":
                    input_devices, output_devices, _, _ = self.get_devices(update=True)
                    self.window["sg_input_device"].update(values=input_devices, value=values['sg_input_device'])
                    self.window["sg_output_device"].update(values=output_devices, value=values['sg_output_device'])
                    sg.popup("Device list has been refreshed.")
                elif event in ("mic_input", "file_input"):
                    self.window['audio_file_row'].update(visible=values["file_input"]); self.window['input_device_col'].update(visible=values["mic_input"])
                elif event == "sg_output_device":
                    if self.audio_processor and self.config.input_source == 'file':
                        self.set_devices(values["sg_input_device"], values["sg_output_device"]); self.audio_processor.change_output_device()
                elif event == "input_audio_path":
                    if self.audio_processor and values["input_audio_path"]:
                        self.audio_processor.request_load_file(values["input_audio_path"])
                elif event == "use_index_file":
                    self.window['index_path_row'].update(visible=values["use_index_file"]); self.window['index_rate_row'].update(visible=values["use_index_file"])
                elif event == "start_vc":
                    if self.set_values(values):
                        self.start_vc()
                        if self.config.input_source == 'file':
                            self.window['file_controls_frame'].update(visible=True)
                            if values['input_audio_path']: self.audio_processor.request_load_file(self.config.input_audio_path)
                elif event == "stop_vc":
                    self.save(values); self.stop_vc()
                elif event == "play_file":
                    if self.audio_processor: self.audio_processor.play_file()
                elif event == "stop_file":
                    if self.audio_processor: self.audio_processor.stop_file()
                elif event == "loop_audio_file":
                    if self.audio_processor: 
                        self.audio_processor.queue_parameter_update(event, values[event])
                        print(f"[PARAM CHANGE] '{event}' updated to '{values[event]}'")


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
                elif key in values: 
                    setattr(self.config, key, values[key])
            
            profile = "Default"
            if values['m2f']: profile = "Male to Female"
            elif values['f2m']: profile = "Female to Male"
            elif values['f2f']: profile = "Female to Female"
            elif values['m2m']: profile = "Male to Male"
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
                print(f"!!! FAILED TO INITIALIZE RVC CLASS !!!\n{traceback.format_exc()} "); sg.popup(f"Failed to initialize RVC: {e}")

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
