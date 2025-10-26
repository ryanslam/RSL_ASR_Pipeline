import torch
import numpy as np
import torchaudio.transforms as T

TARGET_SAMPLE_RATE = 16000


class AudioResampler:
    def __init__(self, dev_sr: int = None, target_sr: int = TARGET_SAMPLE_RATE):
        self.device_sr = None
        self.target_sr = None
        self.resampler = None

        if dev_sr and target_sr:
            self.init_resampler(self.device_sr, self.target_sr)

    def init_resampler(self, dev_sr, target_sr):
        self.device_sr = dev_sr
        self.target_sr = target_sr
        if self.device_sr != self.target_sr:
            self.resampler = T.Resample(
                orig_freq=self.device_sr, new_freq=self.target_sr
            )
            print(f"[Info] Resampling from {self.device_sr} Hz -> {self.target_sr} Hz")
        else:
            self.resampler = None

    def process(self, block: np.ndarray) -> torch.Tensor:
        audio_tensor = torch.from_numpy(block.astype(np.float32))
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=1)
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / 32768.0
        if self.resampler:
            audio_tensor = self.resampler(audio_tensor)
        return audio_tensor
