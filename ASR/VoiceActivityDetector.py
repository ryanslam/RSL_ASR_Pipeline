import torch
import torchaudio
import numpy as np
from torchaudio.transforms import Resample

class VoiceActivityDetector:

    def __init__(self, **kwargs):
        vad_init = False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model, utils = torch.hub.load(repo_or_dir=kwargs['repo'],
                                        model=kwargs['model'],
                                        force_reload=True)

            self.sample_rate = kwargs['sample_rate']
            self.get_speech_timestamps = utils[0]
            self.model = model.to(self.device)
            vad_init = True
        except Exception as e:
            print(f"Unable to initialize VAD model: \n\t{e}")
        finally:
            print(f"VAD initialization status: {vad_init}")

    def _audio_to_tensor(self, audio_data, target_sample_rate):
        audio_tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
        
        # Normalize to range [-1, 1]
        audio_tensor /= 32768.0

        print(f"Audio tensor shape: {audio_tensor.shape}")

        if self.sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=self.sample_rate, new_freq=target_sample_rate)
            audio_tensor = resampler(audio_tensor)
            print(f"Resampled tensor shape: {audio_tensor.shape}")

        return audio_tensor

    def process_audio(self, audio_data, target_sample_rate=16000):
        audio_tensor = self._audio_to_tensor(audio_data, target_sample_rate)
        print(f"Processing audio with shape: {audio_tensor.shape}")
        
        timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=target_sample_rate)
        print(f"Timestamps: {timestamps}")
        
        return timestamps