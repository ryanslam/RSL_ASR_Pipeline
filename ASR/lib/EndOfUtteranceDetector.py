import logging

try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    MODEL = load_silero_vad()
except ImportError:
    logging.warning("silero_vad library not found, defaulting to torch.")
    import torch

    MODEL = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )

REQUIRED_SAMPLE_RATE = 16000
REQUIRED_BLOCK_SIZE = 512


class EndOfUtteranceDetector:
    def __init__(self):
        self.model = MODEL

    def _resample_audio(self, audio, orig_sr, target_sr):
        try:
            import torch
            import torchaudio
        except ImportError:
            raise ImportError(
                "torchaudio is required for resampling. Please install it via 'pip install torchaudio'."
            )

    def is_speech(self):
        pass
