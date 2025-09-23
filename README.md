# GKxMangio RVC Mod UI Complex Parameter

This is a modified version of the Mangio RVC project, focusing on a streamlined real-time voice conversion experience with an intuitive graphical user interface (GUI). This fork introduces a comprehensive suite of features for advanced pitch control, formant shifting, and creative audio effects, along with significant code refactoring for improved stability and performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/glickko/GKxMangio-RVC-Mod-UI/main/ss1.jpg" alt="image">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/glickko/GKxMangio-RVC-Mod-UI/main/ss2.jpg" alt="image">
</p>


## 🔧 Installation

These files are designed to be a **direct replacement** for the corresponding files in an existing Mangio RVC folder.

1.  **Download the Modified Files**:
    * Download `gui_v1.py`, `rvc_for_realtime.py`, and `audio_processor.py` from this repository. or full https://huggingface.co/glickko/GKxMangioRVC-Mod/resolve/main/GKxMangio-RVC-v23.7.0.7z?download=true , and use virtfullline https://www.mediafire.com/file/15u798ajzqq8zy6/vac470full.rar/file , set all in sound windows settings for input and output device is have same samplerate [eg: 48000, 1channel]

2.  **Replace the Original Files**:
    * Navigate to your main Mangio RVC installation folder.
    * Replace the existing `gui_v1.py` and `rvc_for_realtime.py` with the new files you downloaded.
    * Place `audio_processor.py` in the same main folder.
    * Backup your ori guiv1.py and rvc realtime.py files first
    

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
    go-realtime-gui.bat
    ```
    *First time open this will load for 5-10 minutes.
2.  **First time will get bug crack etc when starting new voice, its because old cache session from Mangio still there:
    * You need to reset all param in every tab, after that save as new profile config. then start (this will still cracking etc) but after starting voice infer close app and start           again.
3.  **Every parameter have their tools tip:
    * You need to found your best config, this is for advanced user who want experiment on get the best result output like me.
---

## 🙏 Acknowledgements

* This project is a modification of the original **Mangio-RVC** project. All credit for the core RVC technology and implementation goes to its original creators.
* This UI was built using **PySimpleGUI**.
* Core audio processing and effects rely on **sounddevice**, **librosa**, 'PyTorch', and 'pedalboard'.

## 📄 License

This project is open-source and free to use and modify.




