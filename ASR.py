from "./lib/Transcription.py" import Transcriber, TranscriberBuilder
from "./lib/AudioInputDevices.py" import AudioListener, InputDevicesBuilder

class ASRSystem:
    def __init__(self, vad: VAD, transcriber: Transcriber, audio_listener: AudioListener):
        pass