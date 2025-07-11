import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

if __name__ == "__main__":
    import numpy as np
    import traceback, re
    import json
    import PySimpleGUI as sg
    import sounddevice as sd
    import librosa, torch, time, threading, webbrowser
    from rvc_for_realtime import RVC
    from config import Config
    from audio_processor import AudioProcessor # New Import

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
            # Added Discord effects
            self.enable_discord_effects: bool = False
            self.discord_proximity: float = 1.0
            self.discord_noise: float = -80.0
            self.discord_quality: float = 1.0
            # Separated Phone and Saturation effects
            self.enable_phone_effect: bool = False
            self.enable_saturation_effect: bool = False
            self.saturation_threshold_hz: float = 800.0
            self.saturation_drive_db: float = 6.0
            # Added cave effect
            self.enable_cave_effect: bool = False
            self.cave_delay_time: float = 250.0
            self.cave_feedback: float = 0.4
            self.cave_mix: float = 0.5
            # Added Low Frequency Dampening
            self.enable_low_freq_dampening: bool = False
            self.low_freq_dampening_threshold_hz: float = 100.0
            self.low_freq_dampening_level_db: float = -6.0
            # Added Dynamic Proximity Effect
            self.enable_dynamic_proximity: bool = False
            self.dynamic_proximity_strength: float = 0.5
            self.dynamic_proximity_room_size: float = 0.2
            # Diagnostics
            self.enable_cmd_diagnostics: bool = False
            # Added device attribute for AudioProcessor
            self.device = device


    class GUI:
        def __init__(self) -> None:
            self.config = GUIConfig()
            self.audio_processor: AudioProcessor = None
            self.crossover_notes_list = self.get_crossover_notes()
            self.launcher()

        def get_crossover_notes(self):
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            formatted_notes = []
            for octave in range(2, 6):
                for note in note_names:
                    freq = 440.0 * (2.0 ** ((octave - 4) + (note_names.index(note) - 9) / 12.0))
                    
                    note_type = "Low"
                    if octave == 4:
                        note_type = "Mid"
                    elif octave > 4:
                        note_type = "High"
                    
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
            # Consolidating all default values here
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
                "formant_shift": 1.0, "timbre": 1.0,
                "use_split_pitch_correction": False, "split_pitch_crossover": "C4",
                "low_pitch_strength": 0.3, "low_pitch_max_adjustment": 2.0,
                "high_pitch_strength": 0.3, "high_pitch_max_adjustment": 2.0,
                "use_reverb": False, "reverb_room_size": 0.5, "reverb_damping": 0.5,
                "reverb_wet_level": 0.33, "reverb_dry_level": 0.4,
                "enable_discord_effects": False, "discord_proximity": 1.0, 
                "discord_noise": -80.0, "discord_quality": 1.0,
                "enable_phone_effect": False, 
                "enable_saturation_effect": False, "saturation_threshold_hz": 800.0, "saturation_drive_db": 6.0,
                "enable_cave_effect": False, "cave_delay_time": 250.0, "cave_feedback": 0.4,
                "cave_mix": 0.5,
                "enable_low_freq_dampening": False, "low_freq_dampening_threshold_hz": 100.0, "low_freq_dampening_level_db": -6.0,
                "enable_dynamic_proximity": False, "dynamic_proximity_strength": 0.5, "dynamic_proximity_room_size": 0.2,
                "enable_cmd_diagnostics": False
            }

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices(update=True)
            try:
                with open("values1.json", "r") as j:
                    data = json.load(j)
                    default_data = self.get_default_settings(input_devices, output_devices)
                    for key, value in default_data.items():
                        if key not in data:
                            data[key] = value
            except:
                data = self.get_default_settings(input_devices, output_devices)
                with open("values1.json", "w") as j:
                    json.dump(data, j, indent=4)
            return data
        
        def launcher(self):
            data = self.load()
            sg.theme("DarkGrey9")
            input_devices, output_devices, _, _ = self.get_devices(update=False)
            samplerate_options = [16000, 32000, 40000, 44100, 48000, 96000]
            
            crossover_notes = self.crossover_notes_list
            default_display_crossover = self.get_display_string_from_note(data.get("split_pitch_crossover", "C4"))

            use_index_file = data.get("use_index_file", True)
            current_profile = data.get("voice_profile", "Default")
            use_shout_dampening = data.get("use_shout_dampening", False)
            auto_pitch_correction_enabled = data.get("auto_pitch_correction", False)
            use_split_pitch_correction = data.get("use_split_pitch_correction", False)
            use_reverb = data.get("use_reverb", False)
            enable_discord_effects = data.get("enable_discord_effects", False)
            enable_phone_effect = data.get("enable_phone_effect", False)
            enable_saturation_effect = data.get("enable_saturation_effect", False)
            enable_cave_effect = data.get("enable_cave_effect", False)
            enable_low_freq_dampening = data.get("enable_low_freq_dampening", False)
            enable_dynamic_proximity = data.get("enable_dynamic_proximity", False)


            # --- LAYOUT DEFINITIONS FOR TABS ---

            # Tab 1: Main Settings
            main_tab_layout = [
                [sg.Text("Response Threshold"), sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -30), expand_x=True, tooltip="Noise gate. The voice changer will only activate when your input volume is above this level (in dB).")],
                [sg.Text("Pitch Setting"), sg.Slider(range=(-24, 24), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), expand_x=True, tooltip="Global pitch shift in semitones (12 semitones = 1 octave).")],
                [sg.pin(sg.Column([[sg.Text("Index Rate"), sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0.3), expand_x=True, tooltip="How much of the original model's timbre to replace with the timbre from the .index file. 0 = No effect, 1 = Full effect.") ]], key='index_rate_row', visible=use_index_file))],
                [sg.Text("Input Volume"), sg.Slider(range=(0.0, 2.0), key="input_volume", resolution=0.01, orientation="h", default_value=data.get("input_volume", 1.0), expand_x=True, tooltip="Adjust the volume of your microphone input.")],
                [sg.Text("Output Volume (dB)"), sg.Slider(range=(-60, 12), key="output_volume", resolution=1, orientation="h", default_value=data.get("output_volume", 0.0), expand_x=True, tooltip="Adjust the final output volume.")],
                [sg.Frame("Voice Profile", [[
                    sg.Radio("Default", "voice_profile", key="default", default=current_profile == "Default", enable_events=True, tooltip="No special tweaks are applied."),
                    sg.Radio("M->F", "voice_profile", key="m2f", default=current_profile == "Male to Female", enable_events=True, tooltip="Enables Auto Pitch Correction targeting a female pitch range."), 
                    sg.Radio("F->M", "voice_profile", key="f2m", default=current_profile == "Female to Male", enable_events=True, tooltip="Enables Auto Pitch Correction targeting a male pitch range."), 
                    sg.Radio("F->F", "voice_profile", key="f2f", default=current_profile == "Female to Female", enable_events=True), 
                    sg.Radio("M->M", "voice_profile", key="m2m", default=current_profile == "Male to Male", enable_events=True)
                ]], expand_x=True)],
                [sg.pin(sg.Column([[sg.Checkbox("Auto Pitch Correction", key="auto_pitch_correction", default=auto_pitch_correction_enabled, tooltip="Automatically adjusts your pitch to better match the target profile. (See Auto Pitch tab)", enable_events=True)]], key="auto_pitch_checkbox_column", visible=current_profile in ["Male to Female", "Female to Male"]))],
                [sg.Text("Pitch Algorithm: rmvpe (recommended)", tooltip="The algorithm used to detect the pitch of your voice. RMVPE is recommended for quality and speed.")]
            ]

            # Tab 2: Model & Devices
            model_layout = [
                [sg.Checkbox("Use Index File", key="use_index_file", default=use_index_file, enable_events=True, tooltip="Use a .index file to improve timbre similarity. Requires more VRAM.")],
                [sg.Input(default_text=data.get("pth_path", ""), key="pth_path", expand_x=True), sg.FileBrowse("Select .pth file", initial_folder=os.path.join(os.getcwd(), "weights"), file_types=((". pth"),), tooltip="Path to the main voice model file.")],
                [sg.pin(sg.Column([[sg.Input(default_text=data.get("index_path", ""), key="index_path", expand_x=True), sg.FileBrowse("Select .index file", initial_folder=os.path.join(os.getcwd(), "logs"), file_types=((". index"),), tooltip="Path to the feature index file.") ]], key='index_path_row', visible=use_index_file))]
            ]
            devices_layout = [
                [sg.Text("Input Device"), sg.Combo(input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), expand_x=True, tooltip="Select your microphone.")],
                [sg.Text("Output Device"), sg.Combo(output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), expand_x=True, tooltip="Select your speakers or headphones.")],
                [sg.Text("Device Sample Rate"), sg.Combo(samplerate_options, key="device_samplerate", default_value=data.get("device_samplerate", 48000), expand_x=True, tooltip="The sample rate of your audio devices."), sg.Button("Refresh", key="refresh_devices", tooltip="Refresh the list of audio devices.")]
            ]
            model_devices_tab_layout = [
                [sg.Frame("Load Model", model_layout, expand_x=True)],
                [sg.Frame("Audio Devices", devices_layout, expand_x=True)]
            ]

            # Tab 3: Effects (Scrollable Layout)
            formant_timbre_frame = sg.Frame("Formant & Timbre", [
                [sg.Text("Formant Shift"), sg.Slider(range=(0.0, 2.0), key="formant_shift", resolution=0.01, orientation="h", default_value=data.get("formant_shift", 1.0), expand_x=True, tooltip="Adjusts the formants of the voice.")],
                [sg.Text("Timbre"), sg.Slider(range=(0.0, 2.0), key="timbre", resolution=0.01, orientation="h", default_value=data.get("timbre", 1.0), expand_x=True, tooltip="Adjusts the timbre of the voice.")],
            ], expand_x=True)

            low_freq_dampening_frame = sg.Frame("Low Freq Dampening", [
                [sg.Checkbox("Enable", key="enable_low_freq_dampening", default=enable_low_freq_dampening, enable_events=True, tooltip="Lowers volume when pitch is very low.")],
                [sg.pin(sg.Column([
                    [sg.Text("Threshold (Hz)"), sg.Slider(range=(50, 200), key="low_freq_dampening_threshold_hz", resolution=1, orientation="h", default_value=data.get("low_freq_dampening_threshold_hz", 100.0), expand_x=True)],
                    [sg.Text("Dampen (dB)"), sg.Slider(range=(-24.0, 0.0), key="low_freq_dampening_level_db", resolution=0.5, orientation="h", default_value=data.get("low_freq_dampening_level_db", -6.0), expand_x=True)],
                ], key='low_freq_dampening_column', visible=enable_low_freq_dampening))]
            ], expand_x=True)

            saturation_frame = sg.Frame("Saturation", [
                [sg.Checkbox("Enable", key="enable_saturation_effect", default=enable_saturation_effect, enable_events=True, tooltip="Adds distortion when pitch is high.")],
                [sg.pin(sg.Column([
                    [sg.Text("Threshold (Hz)"), sg.Slider(range=(100, 2000), key="saturation_threshold_hz", resolution=10, orientation="h", default_value=data.get("saturation_threshold_hz", 800.0), expand_x=True)],
                    [sg.Text("Drive (dB)"), sg.Slider(range=(0.0, 24.0), key="saturation_drive_db", resolution=0.5, orientation="h", default_value=data.get("saturation_drive_db", 6.0), expand_x=True)],
                ], key='saturation_column', visible=enable_saturation_effect))]
            ], expand_x=True)
            
            dynamic_proximity_frame = sg.Frame("Dynamic Smartphone Proximity", [
                [sg.Checkbox("Enable", key="enable_dynamic_proximity", default=enable_dynamic_proximity, enable_events=True, tooltip="Simulates moving a smartphone mic around while talking.")],
                [sg.pin(sg.Column([
                    [sg.Text("Strength"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_strength", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_strength", 0.5), expand_x=True, tooltip="How dramatic the effect is.")],
                    [sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="dynamic_proximity_room_size", resolution=0.01, orientation="h", default_value=data.get("dynamic_proximity_room_size", 0.2), expand_x=True, tooltip="The reverb of the simulated room.")],
                ], key='dynamic_proximity_column', visible=enable_dynamic_proximity))]
            ], expand_x=True)

            discord_effects_frame = sg.Frame("Discord Effects", [
                [sg.Checkbox("Enable", key="enable_discord_effects", default=enable_discord_effects, enable_events=True, tooltip="Simulate Discord call effects.")],
                [sg.pin(sg.Column([
                    [sg.Text("Proximity"), sg.Slider(range=(0.0, 1.0), key="discord_proximity", resolution=0.01, orientation="h", default_value=data.get("discord_proximity", 1.0), expand_x=True)],
                    [sg.Text("Noise (dB)"), sg.Slider(range=(-80.0, 12.0), key="discord_noise", resolution=0.5, orientation="h", default_value=data.get("discord_noise", -80.0), expand_x=True)],
                    [sg.Text("Quality"), sg.Slider(range=(0.0, 1.0), key="discord_quality", resolution=0.01, orientation="h", default_value=data.get("discord_quality", 1.0), expand_x=True)],
                ], key='discord_effects_column', visible=enable_discord_effects))]
            ], expand_x=True)
            
            cave_effect_frame = sg.Frame("Cave/Large Room Echo", [
                [sg.Checkbox("Enable", key="enable_cave_effect", default=enable_cave_effect, enable_events=True, tooltip="Simulates a large, echoey space.")],
                [sg.pin(sg.Column([
                    [sg.Text("Delay (ms)"), sg.Slider(range=(50, 1000), key="cave_delay_time", resolution=10, orientation="h", default_value=data.get("cave_delay_time", 250.0), expand_x=True)],
                    [sg.Text("Feedback"), sg.Slider(range=(0.0, 0.9), key="cave_feedback", resolution=0.01, orientation="h", default_value=data.get("cave_feedback", 0.4), expand_x=True)],
                    [sg.Text("Mix"), sg.Slider(range=(0.0, 1.0), key="cave_mix", resolution=0.01, orientation="h", default_value=data.get("cave_mix", 0.5), expand_x=True)],
                ], key='cave_effect_column', visible=enable_cave_effect))]
            ], expand_x=True)
            
            reverb_frame = sg.Frame("Reverb", [
                [sg.Checkbox("Enable", key="use_reverb", default=use_reverb, enable_events=True, tooltip="Add a simple reverb.")],
                [sg.pin(sg.Column([
                    [sg.Text("Room Size"), sg.Slider(range=(0.0, 1.0), key="reverb_room_size", resolution=0.01, orientation="h", default_value=data.get("reverb_room_size", 0.5), expand_x=True)],
                    [sg.Text("Damping"), sg.Slider(range=(0.0, 1.0), key="reverb_damping", resolution=0.01, orientation="h", default_value=data.get("reverb_damping", 0.5), expand_x=True)],
                    [sg.Text("Wet Level"), sg.Slider(range=(0.0, 1.0), key="reverb_wet_level", resolution=0.01, orientation="h", default_value=data.get("reverb_wet_level", 0.33), expand_x=True)],
                    [sg.Text("Dry Level"), sg.Slider(range=(0.0, 1.0), key="reverb_dry_level", resolution=0.01, orientation="h", default_value=data.get("reverb_dry_level", 0.4), expand_x=True)],
                ], key='reverb_settings_column', visible=use_reverb))]
            ], expand_x=True)

            phone_effect_frame = sg.Frame("Phone Effect", [
                [sg.Checkbox("Enable", key="enable_phone_effect", default=enable_phone_effect, tooltip="Simulates a phone call by filtering the audio.")]
            ], expand_x=True)

            effects_col_layout = [
                [formant_timbre_frame],
                [dynamic_proximity_frame],
                [low_freq_dampening_frame],
                [saturation_frame],
                [discord_effects_frame],
                [cave_effect_frame],
                [reverb_frame],
                [phone_effect_frame],
                [sg.Sizer(0, 800)]
            ]
            
            effects_tab_layout = [[
                sg.Column(effects_col_layout, scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)
            ]]


            # Tab 4: Auto Pitch
            advanced_pitch_settings_layout = [
                [sg.Text("Pitch Crossover Note"), sg.Combo(crossover_notes, key="split_pitch_crossover_display", default_value=default_display_crossover, expand_x=True, readonly=True, tooltip="Select the musical note that acts as a boundary between low and high pitch settings.")],
                [sg.Frame("Low Pitch Settings", [
                    [sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="low_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("low_pitch_strength", 0.3), expand_x=True)],
                    [sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 5.0), key="low_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("low_pitch_max_adjustment", 2.0), expand_x=True)],
                ], expand_x=True)],
                [sg.Frame("High Pitch Settings", [
                    [sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="high_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("high_pitch_strength", 0.3), expand_x=True)],
                    [sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 5.0), key="high_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("high_pitch_max_adjustment", 2.0), expand_x=True)],
                ], expand_x=True)],
            ]
            auto_pitch_settings_layout = [
                [sg.pin(sg.Column([
                    [sg.Text("Correction Strength"), sg.Slider(range=(0.0, 1.0), key="auto_pitch_strength", resolution=0.01, orientation="h", default_value=data.get("auto_pitch_strength", 0.3), expand_x=True, tooltip="Controls how strongly the pitch is corrected towards the target. Lower is gentler.")],
                    [sg.Text("Max Adjustment (st)"), sg.Slider(range=(0.0, 5.0), key="auto_pitch_max_adjustment", resolution=0.1, orientation="h", default_value=data.get("auto_pitch_max_adjustment", 2.0), expand_x=True, tooltip="The maximum pitch change (in semitones) the auto-pitch can apply.")],
                ], key='global_pitch_settings', visible=not use_split_pitch_correction))],
                [sg.Text("Pitch Stability"), sg.Slider(range=(0.0, 1.0), key="pitch_stability", resolution=0.01, orientation="h", default_value=data.get("pitch_stability", 0.2), expand_x=True, tooltip="Controls pitch smoothing to reduce robotic sound. Higher values = more smoothing, but slower response.")],
                [sg.Checkbox("Enable Shout Dampening", key="use_shout_dampening", default=use_shout_dampening, enable_events=True, tooltip="Enable a feature to prevent unnatural squeaking when shouting.")],
                [sg.pin(sg.Column([[sg.Text("Shout Dampening Strength"), sg.Slider(range=(0.0, 1.0), key="shout_dampening_strength", resolution=0.01, orientation="h", default_value=data.get("shout_dampening_strength", 0.8), expand_x=True, tooltip="How aggressively to lower pitch during shouts. Higher is more aggressive.")]], key='shout_dampening_slider_row', visible=use_shout_dampening))],
                [sg.HorizontalSeparator()],
                [sg.Checkbox("Enable Split Pitch Correction (Advanced)", key="use_split_pitch_correction", default=use_split_pitch_correction, enable_events=True, tooltip="Use separate correction settings for low and high pitch ranges.")],
                [sg.pin(sg.Frame("Split Pitch Correction Settings", advanced_pitch_settings_layout, key="advanced_pitch_frame", expand_x=True, visible=use_split_pitch_correction))]
            ]
            auto_pitch_tab_layout = [
                [sg.pin(sg.Frame("Auto Pitch Settings", auto_pitch_settings_layout, key="auto_pitch_frame", expand_x=True, visible=auto_pitch_correction_enabled and current_profile in ["Male to Female", "Female to Male"]))]
            ]
            
            # Tab 5: Performance
            performance_settings_layout = [
                [sg.Text("Sample Length (s)"), sg.Slider(range=(0.1, 5.0), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 1.0), expand_x=True, tooltip="The length of the audio chunk processed at once. Lower values mean lower latency but may be less stable.")],
                [sg.Text("Hop Length"), sg.Slider(range=(32, 4096), key="hop_length", resolution=32, orientation="h", default_value=data.get("hop_length", 128), expand_x=True, tooltip="Affects feature extraction. Lower values may improve quality at the cost of higher CPU usage.")],
                [sg.Text("Crossfade Length (ms)"), sg.Slider(range=(10, 500), key="crossfade_length", resolution=10, orientation="h", default_value=data.get("crossfade_length", 80.0), expand_x=True, tooltip="The duration of the crossfade between audio chunks to reduce clicking artifacts.")],
                [sg.Text("Extra Inference Time (ms)"), sg.Slider(range=(50, 5000), key="extra_time", resolution=10, orientation="h", default_value=data.get("extra_time", 40.0), expand_x=True, tooltip="The size of the extra audio buffer used for processing. Increase if you hear stuttering.")],
                [sg.Text("SOLA Search (ms)"), sg.Slider(range=(2, 100), key="sola_search_ms", resolution=1, orientation="h", default_value=data.get("sola_search_ms", 10.0), expand_x=True, tooltip="Synchronized Overlap-Add. A technique to find the best point to stitch audio chunks together, avoiding clicks and phasing.")],
                [sg.Text("Precision"), sg.Radio("FP16", "precision", key="fp16", default=data.get("is_half", True), tooltip="Half-precision. Faster performance, lower VRAM usage, slightly lower quality. Requires a compatible GPU."), sg.Radio("FP32", "precision", key="fp32", default=not data.get("is_half", True), tooltip="Full-precision. Higher quality, higher VRAM usage.")],
                [sg.Checkbox("Input Denoise", key="I_noise_reduce", default=data.get("I_noise_reduce", False), tooltip="Apply a simple noise reduction filter to your microphone input."), sg.Checkbox("Output Denoise", key="O_noise_reduce", default=data.get("O_noise_reduce", False), tooltip="Apply a simple noise reduction filter to the final converted audio.")],
                [sg.Checkbox("Enable CMD Diagnostics", key="enable_cmd_diagnostics", default=data.get("enable_cmd_diagnostics", False), tooltip="Prints real-time processing details to the command line for debugging.")],
                [sg.Text("Model Sample Rate:"), sg.Text("N/A", key="sg_samplerate")],
            ]

            # Tab 6: About Me
            aboutme_tab_layout = [[
                sg.Column([
                    [sg.VPush()],
                    [sg.Text("GKxMangio RVC Mod UI", font=("Helvetica", 24, "bold"), justification='center')],
                    [sg.HorizontalSeparator()],
                    [sg.Text("\nDiscord: @glickko", font=("Helvetica", 12), justification='center')],
                    [sg.Text("Visit Github", text_color="#00B0F0", font=("Helvetica", 12, "underline"), enable_events=True, key='-GITHUB_LINK-', tooltip='https://glickko.github.io')],
                    [sg.VPush()]
                ], element_justification='center', expand_x=True)
            ]]

            # --- STATUS BAR ---
            status_bar_layout = [
                sg.Button("Start Audio Conversion", key="start_vc", disabled=False), 
                sg.Button("Stop Audio Conversion", key="stop_vc", disabled=True), 
                sg.Push(), 
                sg.Text("Avg Pitch (Hz):"), 
                sg.Text("0.00", key="avg_hz"),
                sg.Text("Inference Time (ms):"), 
                sg.Text("0", key="infer_time")
            ]

            # --- ASSEMBLE TABS AND FINAL LAYOUT ---
            tab_group = sg.TabGroup([
                [sg.Tab('Main', main_tab_layout),
                 sg.Tab('Model & Devices', model_devices_tab_layout),
                 sg.Tab('Effects', effects_tab_layout, expand_x=True, expand_y=True),
                 sg.Tab('Auto Pitch', auto_pitch_tab_layout),
                 sg.Tab('Performance', performance_settings_layout),
                 sg.Tab('About Me', aboutme_tab_layout)]
            ], expand_x=True, expand_y=True)

            layout = [
                [tab_group],
                [status_bar_layout]
            ]
            
            self.window = sg.Window("GKxMangio RVC Mod UI v1.10.2 (Stable)", layout=layout, finalize=True, resizable=True)
            self.event_handler()

        def event_handler(self):
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    if self.audio_processor:
                        self.audio_processor.stop()
                    sys.exit()

                # Event from AudioProcessor thread to update GUI
                if event == '-UPDATE_STATUS-':
                    total_time_ms, avg_hz = values[event]
                    self.window["infer_time"].update(f"{int(total_time_ms)}")
                    self.window["avg_hz"].update(f"{avg_hz:.2f}")
                    continue

                if event == '-STREAM_ERROR-':
                    sg.popup("Audio stream error! Please check your audio devices and settings, then restart the application.", title="Error")
                    self.stop_vc() # Clean up buttons
                    continue
                
                if event == '-GITHUB_LINK-':
                    webbrowser.open('https://glickko.github.io')
                    continue

                if event == "refresh_devices":
                    input_devices, output_devices, _, _ = self.get_devices(update=True)
                    current_input = values['sg_input_device']
                    current_output = values['sg_output_device']
                    self.window["sg_input_device"].update(values=input_devices, value=current_input if current_input in input_devices else '')
                    self.window["sg_output_device"].update(values=output_devices, value=current_output if current_output in output_devices else '')
                    sg.popup("Device list has been refreshed.")

                if event == "use_index_file":
                    self.window['index_path_row'].update(visible=values["use_index_file"])
                    self.window['index_rate_row'].update(visible=values["use_index_file"])

                if event in ["default", "m2f", "f2m", "f2f", "m2m"]:
                    is_auto_pitch_profile = values["m2f"] or values["f2m"]
                    self.window["auto_pitch_checkbox_column"].update(visible=is_auto_pitch_profile)
                    self.window["auto_pitch_frame"].update(visible=is_auto_pitch_profile and values["auto_pitch_correction"])
                
                if event == "auto_pitch_correction":
                    is_enabled = values["auto_pitch_correction"]
                    is_auto_pitch_profile = values["m2f"] or values["f2m"]
                    self.window["auto_pitch_frame"].update(visible=is_enabled and is_auto_pitch_profile)
                
                if event == "use_split_pitch_correction":
                    is_split_enabled = values["use_split_pitch_correction"]
                    self.window["global_pitch_settings"].update(visible=not is_split_enabled)
                    self.window["advanced_pitch_frame"].update(visible=is_split_enabled)

                if event == "use_shout_dampening":
                    self.window["shout_dampening_slider_row"].update(visible=values["use_shout_dampening"])
                
                if event == "use_reverb":
                    self.window["reverb_settings_column"].update(visible=values["use_reverb"])

                if event == "enable_discord_effects":
                    self.window["discord_effects_column"].update(visible=values["enable_discord_effects"])
                
                if event == "enable_saturation_effect":
                    self.window["saturation_column"].update(visible=values["enable_saturation_effect"])
                
                if event == "enable_cave_effect":
                    self.window["cave_effect_column"].update(visible=values["enable_cave_effect"])
                
                if event == "enable_low_freq_dampening":
                    self.window["low_freq_dampening_column"].update(visible=values["enable_low_freq_dampening"])

                if event == "enable_dynamic_proximity":
                    self.window["dynamic_proximity_column"].update(visible=values["enable_dynamic_proximity"])

                if event == "start_vc":
                    profile = "Default"
                    if values['m2f']: profile = "Male to Female"
                    elif values['f2m']: profile = "Female to Male"
                    elif values['f2f']: profile = "Female to Female"
                    elif values['m2m']: profile = "Male to Male"
                    values['voice_profile'] = profile
                    
                    values["split_pitch_crossover"] = self.get_note_from_display_string(values["split_pitch_crossover_display"])
                    
                    temp_values = values.copy()
                    del temp_values["split_pitch_crossover_display"]
                    temp_values["formant_shift"] = values["formant_shift"]
                    temp_values["timbre"] = values["timbre"]

                    with open("values1.json", "w") as j:
                        json.dump(temp_values, j, indent=4)
                    
                    if self.set_values(values):
                        self.start_vc()

                if event == "stop_vc":
                    self.stop_vc()

            self.window.close()

        def stop_vc(self):
            if self.audio_processor:
                self.audio_processor.stop()
                self.audio_processor = None
            
            # Reset UI elements
            self.window["sg_samplerate"].update("N/A")
            self.window["start_vc"].update(disabled=False)
            self.window["stop_vc"].update(disabled=True)
            self.window["infer_time"].update("0")
            self.window["avg_hz"].update("0.00")
        
        def set_values(self, values):
            if not values["pth_path"].strip():
                sg.popup("Please select a .pth file")
                return False
            
            self.config.use_index_file = values["use_index_file"]
            if self.config.use_index_file and (not values["index_path"].strip()):
                sg.popup("Please select an .index file or uncheck 'Use Index File'")
                return False

            pattern = re.compile("[\u4e00-\u9fa5]+")
            if pattern.search(values["pth_path"]) or (self.config.use_index_file and pattern.search(values["index_path"])):
                sg.popup("File paths cannot contain non-ASCII characters")
                return False
            
            self.set_devices(values["sg_input_device"], values["sg_output_device"])

            # Explicitly set all config attributes
            self.config.pth_path = values["pth_path"]
            self.config.index_path = values["index_path"]
            self.config.pitch = values["pitch"]
            self.config.block_time = values["block_time"]
            self.config.threhold = values["threhold"]
            self.config.crossfade_time = values["crossfade_length"] / 1000
            self.config.extra_time = values["extra_time"] / 1000
            self.config.I_noise_reduce = values["I_noise_reduce"]
            self.config.O_noise_reduce = values["O_noise_reduce"]
            self.config.index_rate = values["index_rate"] if self.config.use_index_file else 0.0
            self.config.f0method = "rmvpe"
            self.config.is_half = values["fp16"]
            self.config.hop_length = int(values["hop_length"])
            self.config.device_samplerate = int(values["device_samplerate"])
            self.config.sola_search_ms = values["sola_search_ms"] / 1000
            self.config.input_volume = values["input_volume"]
            self.config.output_volume = values["output_volume"]
            self.config.auto_pitch_correction = values["auto_pitch_correction"]
            self.config.pitch_stability = values["pitch_stability"]
            self.config.auto_pitch_strength = values["auto_pitch_strength"]
            self.config.auto_pitch_max_adjustment = values["auto_pitch_max_adjustment"]
            self.config.voice_profile = values["voice_profile"]
            self.config.use_shout_dampening = values["use_shout_dampening"]
            self.config.shout_dampening_strength = values["shout_dampening_strength"]
            self.config.formant_shift = values["formant_shift"]
            self.config.timbre = values["timbre"]
            self.config.use_split_pitch_correction = values["use_split_pitch_correction"]
            self.config.split_pitch_crossover = self.get_note_from_display_string(values["split_pitch_crossover_display"])
            self.config.low_pitch_strength = values["low_pitch_strength"]
            self.config.low_pitch_max_adjustment = values["low_pitch_max_adjustment"]
            self.config.high_pitch_strength = values["high_pitch_strength"]
            self.config.high_pitch_max_adjustment = values["high_pitch_max_adjustment"]
            self.config.use_reverb = values["use_reverb"]
            self.config.reverb_room_size = values["reverb_room_size"]
            self.config.reverb_damping = values["reverb_damping"]
            self.config.reverb_wet_level = values["reverb_wet_level"]
            self.config.reverb_dry_level = values["reverb_dry_level"]
            self.config.enable_discord_effects = values["enable_discord_effects"]
            self.config.discord_proximity = values["discord_proximity"]
            self.config.discord_noise = values["discord_noise"]
            self.config.discord_quality = values["discord_quality"]
            self.config.enable_phone_effect = values["enable_phone_effect"]
            self.config.enable_saturation_effect = values["enable_saturation_effect"]
            self.config.saturation_threshold_hz = values["saturation_threshold_hz"]
            self.config.saturation_drive_db = values["saturation_drive_db"]
            self.config.enable_cave_effect = values["enable_cave_effect"]
            self.config.cave_delay_time = values["cave_delay_time"]
            self.config.cave_feedback = values["cave_feedback"]
            self.config.cave_mix = values["cave_mix"]
            self.config.enable_low_freq_dampening = values["enable_low_freq_dampening"]
            self.config.low_freq_dampening_threshold_hz = values["low_freq_dampening_threshold_hz"]
            self.config.low_freq_dampening_level_db = values["low_freq_dampening_level_db"]
            self.config.enable_dynamic_proximity = values["enable_dynamic_proximity"]
            self.config.dynamic_proximity_strength = values["dynamic_proximity_strength"]
            self.config.dynamic_proximity_room_size = values["dynamic_proximity_room_size"]
            self.config.enable_cmd_diagnostics = values["enable_cmd_diagnostics"]
            
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            print("---STARTING RVC---")
            print("---SETTINGS---")
            for key, value in self.config.__dict__.items():
                print(f"{key}: {value}")
            
            print("---INITIALIZING RVC CLASS---")
            try:
                # Call RVC with all explicit keyword arguments from the config object
                rvc = RVC(
                    key=self.config.pitch,
                    pth_path=self.config.pth_path,
                    index_path=self.config.index_path if self.config.use_index_file else None,
                    index_rate=self.config.index_rate,
                    device=device,
                    auto_pitch_correction=self.config.auto_pitch_correction,
                    pitch_stability=self.config.pitch_stability,
                    auto_pitch_strength=self.config.auto_pitch_strength,
                    auto_pitch_max_adjustment=self.config.auto_pitch_max_adjustment,
                    voice_profile=self.config.voice_profile,
                    use_shout_dampening=self.config.use_shout_dampening,
                    shout_dampening_strength=self.config.shout_dampening_strength,
                    formant_shift=self.config.formant_shift,
                    timbre=self.config.timbre,
                    use_split_pitch_correction=self.config.use_split_pitch_correction,
                    split_pitch_crossover=self.config.split_pitch_crossover,
                    low_pitch_strength=self.config.low_pitch_strength,
                    low_pitch_max_adjustment=self.config.low_pitch_max_adjustment,
                    high_pitch_strength=self.config.high_pitch_strength,
                    high_pitch_max_adjustment=self.config.high_pitch_max_adjustment,
                    use_reverb=self.config.use_reverb,
                    reverb_room_size=self.config.reverb_room_size,
                    reverb_damping=self.config.reverb_damping,
                    reverb_wet_level=self.config.reverb_wet_level,
                    reverb_dry_level=self.config.reverb_dry_level,
                    # Discord Effects
                    enable_discord_effects=self.config.enable_discord_effects,
                    discord_proximity=self.config.discord_proximity,
                    discord_noise=self.config.discord_noise,
                    discord_quality=self.config.discord_quality,
                    # Phone and Saturation Effects
                    enable_phone_effect=self.config.enable_phone_effect,
                    enable_saturation_effect=self.config.enable_saturation_effect,
                    saturation_threshold_hz=self.config.saturation_threshold_hz,
                    saturation_drive_db=self.config.saturation_drive_db,
                    # Cave Effect
                    enable_cave_effect=self.config.enable_cave_effect,
                    cave_delay_time=self.config.cave_delay_time,
                    cave_feedback=self.config.cave_feedback,
                    cave_mix=self.config.cave_mix,
                    # Low Frequency Dampening
                    enable_low_freq_dampening=self.config.enable_low_freq_dampening,
                    low_freq_dampening_threshold_hz=self.config.low_freq_dampening_threshold_hz,
                    low_freq_dampening_level_db=self.config.low_freq_dampening_level_db,
                    # Dynamic Proximity
                    enable_dynamic_proximity=self.config.enable_dynamic_proximity,
                    dynamic_proximity_strength=self.config.dynamic_proximity_strength,
                    dynamic_proximity_room_size=self.config.dynamic_proximity_room_size,
                    # Diagnostics
                    enable_cmd_diagnostics=self.config.enable_cmd_diagnostics
                )
                print("---RVC CLASS INITIALIZED SUCCESSFULLY---")
                self.config.samplerate = rvc.tgt_sr # Set samplerate in config for processor
                self.window["sg_samplerate"].update(f"{rvc.tgt_sr} Hz")
                
                # Create and start the audio processor
                self.audio_processor = AudioProcessor(self.config, rvc, self.window)
                self.audio_processor.start()

                self.window["start_vc"].update(disabled=True)
                self.window["stop_vc"].update(disabled=False)

            except Exception as e:
                print("!!! FAILED TO INITIALIZE RVC CLASS !!!")
                print(traceback.format_exc())
                sg.popup(f"Failed to initialize RVC: {e}")
                return

        def get_devices(self, update: bool = True):
            if update:
                sd._terminate()
                sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            input_devices = [f"{d['name']} ({d['hostapi_name']})" for d in devices if d["max_input_channels"] > 0]
            output_devices = [f"{d['name']} ({d['hostapi_name']})" for d in devices if d["max_output_channels"] > 0]
            input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
            output_devices_indices = [d["index"] for d in devices if d["max_output_channels"] > 0]
            return (input_devices, output_devices, input_devices_indices, output_devices_indices)

        def set_devices(self, input_device, output_device):
            input_devices, _, input_device_indices, _ = self.get_devices(update=False)
            _, output_devices, _, output_device_indices = self.get_devices(update=False)
            try:
                sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
                sd.default.device[1] = output_device_indices[output_devices.index(output_device)]
                print(f"input device:{sd.default.device[0]}:{input_device}")
                print(f"output device:{sd.default.device[1]}:{output_device}")
            except Exception as e:
                sg.popup(f"Error setting devices: {e}")

    gui = GUI()