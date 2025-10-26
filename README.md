# Active Speech Recognition Pipeline Repository.
This repository implements active speech recognition to allow for voice input. The goal of this project is to act as a modular application that detects user speech and transcribes it.
The resulting text will be accessible via a ZMQ Pub/Sub topic.

## Currently, the ASR pipeline leverages the following libs/models:
### Libraries:
  - Sounddevice: Necessary to select audio input device.
  - torch
  - torchaudio
  - numpy
  - ZMQ
  
### Models:
  - Silero-VAD: Used to detect user speech activity.
  - WhisperX: Used to transcribe user speech audio to text.
