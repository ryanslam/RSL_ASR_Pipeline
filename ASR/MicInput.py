import pyaudio
import torch
import numpy as np

class MicInput:

    def __init__(self,
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                chunk_size=1024):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size

    def initialize_audio_stream(self) -> None:
        self.stream = self.audio.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk_size)
    
    def listen(self) -> None|bytes:
        if not self.stream:
            print("No active stream to begin listening!")
            return
        return self.stream.read(self.chunk_size, exception_on_overflow=False)
    
    def raw_to_buffer(self, audio_bytes) -> np.ndarray:
        return np.frombuffer(audio_bytes, dtype=np.int16)

    def stop_listening(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()