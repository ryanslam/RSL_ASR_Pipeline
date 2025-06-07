import torch
import numpy as np
from MicInput import MicInput
from VoiceActivityDetector import VoiceActivityDetector
import time



voice = MicInput(chunk_size=512, sample_rate=16000)
voice.initialize_audio_stream()
VAD = VoiceActivityDetector(repo='snakers4/silero-vad', 
                            model='silero_vad', 
                            sample_rate=voice.sample_rate)

try:
    while True:
        audio_chunk = voice.listen()
        if not audio_chunk:
            continue

        timestamps = VAD.process_audio(audio_chunk)
        if(timestamps):
            print(timestamps)
except KeyboardInterrupt:
    voice.stop_listening()