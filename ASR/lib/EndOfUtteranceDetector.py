import io
import logging
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import time

MODEL, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)

(get_speech_timestamps, _, _, _, _) = utils

REQUIRED_SAMPLE_RATE = 16000
REQUIRED_BLOCK_SIZE = 2048

class EndOfUtteranceDetector:
    def __init__(self, sample_rate=REQUIRED_SAMPLE_RATE, sensitivity_threshold=0.5, min_silence_duration=1.0):
        self.model = MODEL
        self.utils = utils
        
        self.sample_rate = sample_rate
        self.sensitivity_threshold = sensitivity_threshold
        self.min_silence_duration = min_silence_duration
        
        self.active = False
        self.last_speech_time = 0.0
        
    # def _bytes_to_tensor(self, audio_bytes:bytes) -> torch.Tensor:
    #     if isinstance(audio_bytes, bytes):
    #         raise ValueError("Input must be bytes.")
    #     try:
    #         audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    #     except Exception :
    #         audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    #         audio_tensor = torch.from_numpy(audio_np).float() / 32768.0
    #         audio_tensor = audio_tensor.unsqueeze(0)
    #         sample_rate = self.sample_rate
        
    #     # Convert stereo to mono if needed
    #     if audio_tensor.shape[0] > 1:
    #         audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        
    #     if sample_rate != REQUIRED_SAMPLE_RATE:
    #         audio_tensor = self._resample_audio(
    #             audio_tensor, orig_sr=sample_rate, target_sr=self.sample_rate
    #         )
            
    #     return audio_tensor.squeeze(0)
    
    def _array_to_tensor(self, audio_array: np.ndarray, sample_rate) -> torch.Tensor:
        """
        Convert a float32 numpy array (from sounddevice) to a mono torch tensor.
        audio_array: shape (frames, channels) or (frames,)
        """
        # Convert to torch
        audio_tensor = torch.from_numpy(audio_array)
        
        if audio_tensor.ndim > 1 and audio_tensor.shape[1] > 1:
            audio_tensor = audio_tensor.mean(dim=1)
            
        if sample_rate != REQUIRED_SAMPLE_RATE:
            audio_tensor = self._resample_audio(
                audio_tensor, orig_sr=sample_rate, target_sr=REQUIRED_SAMPLE_RATE
            )
            
        audio_tensor = audio_tensor.squeeze()

        return audio_tensor


    def _resample_audio(self, audio:torch.Tensor, orig_sr:int, target_sr:int) -> torch.Tensor:
        try:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled_audio = resampler(audio)
        except ImportError:
            raise ImportError(
                "torchaudio is required for resampling. Please install it via 'pip install torchaudio'."
            )
            
        return resampled_audio
    
    def is_speech(self, audio_tensor) -> list:
        """Determine if the audio tensor contains speech."""
        segments = []
        with torch.no_grad():
            segments = get_speech_timestamps(audio_tensor, self.model, sampling_rate=REQUIRED_SAMPLE_RATE)
        return segments

    def set_valid_blocksize(self, audio_tensor) -> torch.Tensor:        
        if audio_tensor.shape[-1] > REQUIRED_BLOCK_SIZE:
            audio_tensor = audio_tensor[:REQUIRED_BLOCK_SIZE]
        elif audio_tensor.shape[-1] < REQUIRED_BLOCK_SIZE:
            padding = torch.zeros(REQUIRED_BLOCK_SIZE - audio_tensor.shape[-1])
            audio_tensor = torch.cat([audio_tensor, padding])
        
        return audio_tensor
    
    def update(self, audio, sample_rate) -> bool:
        audio_tensor = self._array_to_tensor(audio, sample_rate)
        audio_tensor = self.set_valid_blocksize(audio_tensor)

        segments = self.is_speech(audio_tensor)
        
        if segments:
            self.last_speech_time = self.last_speech_time = segments[-1]['end']
            self.active = True
        else:
            num_samples_since_last_speech = audio_tensor.shape[-1] - self.last_speech_time
            min_silence_samples = int(self.min_silence_duration * REQUIRED_SAMPLE_RATE)
            
            if self.active and num_samples_since_last_speech >= min_silence_samples:
                self.active = False
                print('end of utterance detected')
                return True
        return False