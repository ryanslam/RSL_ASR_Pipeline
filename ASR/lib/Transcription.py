import logging
import whisperx
import torch


class Transcriber:
    def __init__(self, whisper_model):
        self.whisper_model = whisper_model

    def transcribe(self, audio_bytes):
        return self.whisper_model.transcribe(audio_bytes)


class TranscriberBuilder:
    def __init__(self):
        self.device = None
        self.model_size = None
        self.whisper_model = None
        self.vad = None
        self.batch_size = 16
        self.compute_type = "float32"

    def set_vad(self, vad_method="silero"):
        self.vad = vad_method
        return self

    def set_device(self, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        self.device = device
        logging.info(f"WhisperX will use device: {self.device}")

        return self

    def set_whisper_model(self, model):
        if not self.device:
            raise ValueError("Device not set. Please set device before loading model.")
        self.model_size = model
        return self

    def set_compute_type(self, compute_type="float32"):
        self.compute_type = compute_type
        return self

    def build(self):
        whisper_model = whisperx.load_model(
            model=self.model_size,
            device=self.device,
            vad_method=self.vad,
            compute_type=self.compute_type,
        )
        return Transcriber(whisper_model)
