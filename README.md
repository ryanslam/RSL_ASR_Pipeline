# Active Speech Recognition (ASR)
This repository implements an Active Speech Recognition. This allows users to speak into their mic and publish the transcribed text via a ZMQ socket.

ASR is primarily dependent on the following:
- Sounddevice: Microphone selection and audio input stream.
- Silero-VAD: Used to detect voice activity and determine end of utterance (EOU).
- WhisperX: Audio -> text transcription.

# How to use:
1. Install the requirements by running the following `pip3 install -r requirements.txt`
2. Configure ASR settings.
    - Main configuration settings are located in `config/config.yaml`
        - To use another configuration file, pass the file path using `--config` 
    - Override configuration settings using flags. For more information run `python3 main.py --h`
3. Main file is located at `ASR/main.py`. Run this file using `python3 main.py`
4. Begin speaking into the selected microphone.
5. If you need to check if the text is being published, run `Examples/example_sub.py`

# Future Plans:
- Allow configurable VAD and transcription models.
- Create interface for microphone selection/settings.
- Enable different data transportion architectures beyond standard PUB/SUB.