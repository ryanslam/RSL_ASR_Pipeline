import torch
import numpy as np
import librosa

class VoiceActivityDetector:

    def __init__(self, **kwargs):
        vad_init = False
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model, utils = torch.hub.load(repo_or_dir=kwargs['repo'],
                                        model=kwargs['model'],
                                        force_reload=True)

            self.sampling_rate = kwargs['sampling_rate']
            self.get_speech_timestamps = utils[0]
            self.model = model.to(self.device)
            vad_init = True
        except Exception as e:
            print(f"Unable to initialize VAD model: \n\t{e}")
        finally:
            print(f"VAD initialization status: {vad_init}")

    def _audio_to_tensor(self, audio_data, target_sampling_rate):
        np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        # Expected sampling rate for silero is 8000 or 16000
        if self.sampling_rate != target_sampling_rate:
            np_audio = librosa.resample(np_audio.astype(np.float32),
                                          orig_sr=self.sampling_rate,
                                          target_sr=target_sampling_rate)
        np_audio /= 32768.0
        # print(np_audio)
        return torch.from_numpy(np_audio)

    def process_audio(self, audio_data, target_sampling_rate=16000):
        audio_tensor = self._audio_to_tensor(audio_data, target_sampling_rate)
        # print(audio_tensor)
        return self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=target_sampling_rate, return_seconds=True)