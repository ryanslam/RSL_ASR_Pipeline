import torch
from collections import deque

MODEL, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
(get_speech_timestamps, _, _, _, _) = utils
TARGET_SAMPLE_RATE = 16000


class EOUDetector:
    def __init__(self, min_silence_sec=1.0, vad_window_sec=0.5):
        self.min_silence_samples = int(min_silence_sec * TARGET_SAMPLE_RATE)
        self.vad_window_samples = int(vad_window_sec * TARGET_SAMPLE_RATE)

        self.total_samples = 0
        self.last_speech_sample = 0
        self.active = False

        self.vad_buffer = deque(maxlen=self.vad_window_samples)

    def process_block(self, audio_tensor: torch.Tensor) -> bool:
        # Extend buffer + global counters
        self.vad_buffer.extend(audio_tensor.tolist())
        self.total_samples += len(audio_tensor)

        # Always run VAD even if buffer is small
        vad_input = torch.tensor(list(self.vad_buffer))

        segments = get_speech_timestamps(
            vad_input, MODEL, sampling_rate=TARGET_SAMPLE_RATE
        )

        if segments:
            # Speech detected → reset silence counter + activate
            self.last_speech_sample = self.total_samples
            self.active = True
        else:
            # No speech detected → may deactivate only if long enough silence
            silence_duration = self.total_samples - self.last_speech_sample
            if self.active and silence_duration >= self.min_silence_samples:
                self.active = False

        return self.active
