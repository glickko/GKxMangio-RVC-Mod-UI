# GKxMangio RVC Mod UI Complex Parameter


##IMPORTANT
Default profile will not affected by formant shift, timbre, and auto pitch. you need choose other profile when you want to use Auto Timbre, Auto Pitch ETC



This is a modified version of the Mangio RVC project, focusing on a streamlined real-time voice conversion experience with an intuitive graphical user interface (GUI). This fork introduces a comprehensive suite of features for advanced pitch control, formant shifting, and creative audio effects, along with significant code refactoring for improved stability and performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/glickko/GKxMangio-RVC-Mod-UI/main/ss1.jpg" alt="image">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/glickko/GKxMangio-RVC-Mod-UI/main/ss2.jpg" alt="image">
</p>



## 👾 Version Section

V1.xx Section is just only focusing in Advanced Parameter Features, the last stable version is `v1.10.2`, the next section will be in v2.xx focusing on other features like adding input from file or changing parameter in realtime mode without stopping the process, and other..

V2.xx Is advanced features for realtime changing parameters, or adding audio files.

## ✨ Key Features & Modifications

This version adds a significant number of new features and quality-of-life improvements over the base Mangio RVC real-time GUI.

* **Completely Redesigned UI**: The user interface has been overhauled with a modern dark theme and a responsive, scrollable layout to neatly organize the large number of new parameters.
* **"Use Index File" Checkbox**: Dynamically enable or disable the use of a `.index` file without restarting the application. This also hides the associated "Index Rate" slider when not in use.
* **Advanced Pitch Control**: A sophisticated system that dynamically adjusts your voice's pitch to a target range. This includes controls for **Correction Strength**, **Pitch Stability** (to reduce robotic sounds), and **Max Adjustment**.
* **Voice Profiles**: Instead of just raw pitch shifting, you can now select a profile (`Male to Female`, `Female to Male`, etc.) which automatically enables advanced features tailored for that conversion type, all of this profile have their own base Hz, so `experiment` on this will give you good results.
* **Shout Dampening**: A smart feature that prevents your voice from cracking or becoming a high-pitched squeak when you laugh or shout by dynamically lowering the pitch adjustment.
* **Universal Formant & Timbre Controls**: These controls, which affect the character and richness of the voice, are now available for all voice profiles.
* **Split Pitch Correction**: This powerful feature allows you to set two different sets of pitch correction rules based on a **Pitch Crossover Note**, allowing for more nuanced and natural-sounding inflection.
* **Comprehensive Effects Suite**: Integrated a suite of creative audio effects including a high-quality **Reverb**, a **Phone Simulator**, pitch-based **Saturation**, an echoey **Cave Effect**, and a complex **Discord Call Simulator**, all of this effects is focuing to be outputing a `Realistic` like.
* **Dynamic Audio Processing**: Added **Low-Frequency Dampening** to reduce muddiness and a **Dynamic Proximity Effect** that realistically simulates the sound of a microphone moving closer or further away based on your input volume.
* **Real-time Status Display**: The status bar now shows the `Inference Time (ms)` and `Avg Pitch (Hz)` of your raw voice, giving you immediate feedback for performance tuning when using advanced pitch feature splitting.
* **Major Code Refactoring**: The core audio processing logic has been moved into a separate `AudioProcessor` class, making the application more stable and reducing UI freezes.
* **Output Volume in Decibels (dB)**: The output volume is now measured in dB. This helps stabilize the output volume, ensuring different models produce a more consistent loudness.
* **RMVPE as the Sole Pitch Algorithm**: All older pitch-finding algorithms have been removed to streamline the process. RMVPE is used exclusively for its superior speed and accuracy.


## 🔧 Installation

These files are designed to be a **direct replacement** for the corresponding files in an existing Mangio RVC folder.

1.  **Download the Modified Files**:
    * Download `gui_v1.py`, `rvc_for_realtime.py`, and `audio_processor.py` from this repository.

2.  **Replace the Original Files**:
    * Navigate to your main Mangio RVC installation folder.
    * Replace the existing `gui_v1.py` and `rvc_for_realtime.py` with the new files you downloaded.
    * Place `audio_processor.py` in the same main folder.

3.  **Install Dependencies for Advanced Effects**:
    * The advanced effects suite (Reverb, Saturation, Cave, etc.) requires the `pedalboard` library. To install it into Mangio's specific Python environment, open a command prompt or terminal in your Mangio RVC root folder.
    * Run the following command:
        ```bash
        runtime\python.exe -m pip install pedalboard
        ```
    * This ensures the library is installed correctly and accessible by the application.

## 🚀 Quick Start Guide

1.  **Launch the application** as you normally would:
    ```bash
    runtime\python.exe gui_v1.py
    ```
2.  **Load Your Model** (Model & Devices Tab):
    * Click "Select .pth file" to load your main voice model from the `weights` folder.
    * Check "Use Index File" and select your `.index` file from the `logs` folder for better timbre matching, i prefer not use this for better performance.
3.  **Configure Audio Devices** (Model & Devices Tab):
    * Select your microphone from the "Input Device" dropdown.
    * Select your headphones or speakers from the "Output Device" dropdown.
    * Click "Refresh" if your devices don't appear.
4.  **Start Conversion**:
    * Click **"Start Audio Conversion"**. You should hear your voice converted in real-time.
5.  **Tune and Experiment**:
    * Adjust the "Pitch Setting" on the 'Main' tab to change the pitch.
    * Select a "Voice Profile" and enable "Auto Pitch Correction" for more advanced pitch management.
    * Explore the 'Effects' tab to add creative flair to your voice.
    * Click **"Stop Audio Conversion"** when finished.

## 💡 Recommendations

* **Audio Routing**: For feeding your converted voice into applications like Discord or OBS, use a virtual audio cable. I recommend "VB-CABLE" or VAC Full from my page tools `https://glickko.github.io/tools.html` and search for MIC label.
* **Model Training**: For best results with this GUI, consider training your RVC models using the `SeoulStreamingStation/KLM49_HFG` pre-trained model as a base, which can be found on Hugging Face.
* **Experimentation is Key**: The best settings will vary greatly depending on your voice, your model, and your hardware. Take the time to experiment with all the parameters to find what works best for you.

---

## 🎛️ The Ultimate Parameter Guide

This guide provides a detailed explanation of every parameter in the user interface and how it affects the final audio output.

### Main Tab

* **Response Threshold**: A noise gate. Any sound from your microphone that is quieter than this threshold will be treated as silence. This is useful for eliminating background noise. Set this just above your ambient noise level. A good starting point is `-45` dB, RECOM: `DON'T USE THIS` use in `-60` for `TURN OFF`.
* **Pitch Setting**: A static, global pitch shift applied to your voice, measured in semitones (`+12` = one octave up). This is the primary control for changing your pitch.
* **Index Rate**: Controls the influence of the `.index` file on the final timbre. A higher rate makes the model try harder to match the timbre from the training data. A value between `0.5` and `0.8` is often a good balance, `off` or `0` is the best choice hahaha.
* **Input/Output Volume**: Simple volume controls for the microphone input and the final audio output.
* **Voice Profile**: A set of pre-configured logic for different voice conversion goals. `Male to Female` and `Female to Male` profiles activate the Auto Pitch Correction system.

### Auto Pitch Tab

* **Correction Strength**: Controls how aggressively the system pulls your current pitch towards the target neutral pitch (e.g., 190Hz for M2F). Keep this low, typically between `0.2` and `0.4`, to maintain natural-sounding speech.
* **Pitch Stability**: A smoothing factor for the pitch adjustments. This is the key to reducing robotic sounds. For a natural voice, set this relatively high, between `0.6` and `0.8`.
* **Enable Split Pitch Correction (Advanced)**: This powerful feature allows you to apply different pitch correction rules to the lower and higher parts of your voice, separated by the **Pitch Crossover Note**. This is useful for creating expressive and dynamic speech where your normal speaking voice is corrected differently from your shouts or high-pitched tones.

### Effects Tab

#### Formant & Timbre
* **Formant Shift**: Changes the resonant characteristics of the voice, which we perceive as its timbre or "character." This is critical for making a voice sound authentically male or female without changing the pitch. For M2F, try values between `1.05` and `1.15`. For F2M, try `0.85` to `0.95`.
* **Timbre**: A fine-tuning control for the voice's richness. Higher values sound "fuller," while lower values sound "thinner."

#### Dynamic Smartphone Proximity
* **What it is**: Simulates the sound of a smartphone mic moving closer or further away from you based on how loudly you speak.
* **How it affects the output**: Louder input results in more bass and less room reverb (closer mic). Softer input results in a thinner sound with more reverb (further mic). `Strength` controls how dramatic the effect is. `Room Size` adjusts the reverb character.

#### Low Freq Dampening
* **What it is**: Automatically lowers the volume when your voice pitch is very low.
* **How it affects the output**: This prevents deep, "muddy" or rumbling sounds. `Threshold (Hz)` sets the pitch below which the effect activates, and `Dampen (dB)` sets the amount of volume reduction.

#### Saturation
* **What it is**: Adds a warm distortion or "overdrive" effect when your voice pitch is high.
* **How it affects the output**: Useful for adding grit or intensity to shouts. `Threshold (Hz)` sets the pitch above which saturation kicks in. `Drive (dB)` controls the amount of distortion.

#### Discord Effects
* **What it is**: A collection of effects to simulate the sound of a Discord voice call.
* **How it affects the output**:
    * `Proximity`: Simulates mic distance. Lower values sound further away (thinner, less volume).
    * `Noise (dB)`: Mixes in adjustable white noise to simulate a noisy mic.
    * `Quality`: Simulates bitrate. Lower values cut high frequencies, making the voice sound more compressed and "digital."

#### Cave/Large Room Echo
* **What it is**: Simulates a large, echoey space like a cave.
* **How it affects theoutput**:
    * `Delay (ms)`: The time between each echo.
    * `Feedback`: How many times the sound echoes.
    * `Mix`: The volume of the echo effect.

#### Reverb
* **What it is**: A standard reverb effect to simulate an acoustic space.
* **How it affects the output**: `Room Size` controls the size of the simulated room. `Damping` absorbs high frequencies in the echoes. `Wet Level` is the volume of the reverb, and `Dry Level` is the volume of your clean voice. For natural speech, use very subtle settings.

#### Phone Effect
* **What it is**: Simulates the narrow, slightly distorted sound of a phone call.
* **How it affects the output**: This applies a band-pass filter, cutting the very low and very high frequencies of your voice.

### Performance Tab

* **Sample Length (s)**: The size of the audio chunk processed at once. This is a direct trade-off between **latency** and **stability**. Lower values mean lower delay but can cause stuttering. Higher values increase delay but result in smoother audio.
* **Hop Length**: Affects feature extraction. Lower values may improve quality but use more CPU. Higher values can improve performance at a slight quality cost.
* **Crossfade Length (ms)**: The time taken to fade one processed audio chunk into the next. This is essential for eliminating clicking sounds between chunks.
* **Extra Inference Time (ms)**: An additional buffer of audio fed to the model for context. Increasing this can improve the naturalness of the speech if words are getting cut off.
* **SOLA Search (ms)**: Synchronized Overlap-Add. A technique to find the best point to stitch audio chunks together. Leave at default unless you hear phasing or strange artifacts.
* **Precision**: The numerical precision used for calculations on the GPU. `FP16` (half-precision) is much faster and uses less VRAM, with a minimal quality difference. Use this unless you have a high-end GPU.
* **Input/Output Denoise**: Applies a simple noise reduction filter. Use only if you have persistent background hiss, as it can slightly alter voice quality.

---

## 🙏 Acknowledgements

* This project is a modification of the original **Mangio-RVC** project. All credit for the core RVC technology and implementation goes to its original creators.
* This UI was built using **PySimpleGUI**.
* Core audio processing and effects rely on **sounddevice**, **librosa**, 'PyTorch', and 'pedalboard'.

## 📄 License

This project is open-source and free to use and modify.

