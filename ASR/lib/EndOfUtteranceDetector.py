import torch
import numpy as np
from collections import deque

MODEL, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
(get_speech_timestamps, _, _, _, _) = utils
TARGET_SAMPLE_RATE = 16000


class EOUDetector:
    def __init__(self, min_silence_sec=1.0, vad_window_sec=0.3, max_buffer_sec=10):
        self.min_silence_samples = int(min_silence_sec * TARGET_SAMPLE_RATE)
        self.vad_window_samples = int(vad_window_sec * TARGET_SAMPLE_RATE)
        self.max_buffer_samples = int(max_buffer_sec * TARGET_SAMPLE_RATE)

        self.last_speech_sample = 0
        self.total_samples = 0
        self.active = False

        self.buffer = deque(maxlen=self.max_buffer_samples)
        self.vad_buffer = deque(maxlen=self.vad_window_samples)

    def process_block(self, audio_tensor: torch.Tensor):
        # update buffers
        self.buffer.extend(audio_tensor.tolist())
        self.vad_buffer.extend(audio_tensor.tolist())
        self.total_samples += len(audio_tensor)

        # only run VAD if sliding window is full
        if len(self.vad_buffer) >= self.vad_window_samples:
            vad_input = torch.tensor(list(self.vad_buffer))
            segments = get_speech_timestamps(
                vad_input, MODEL, sampling_rate=TARGET_SAMPLE_RATE
            )

            if segments:
                self.last_speech_sample = (
                    self.total_samples - len(vad_input) + segments[-1]["end"]
                )
                self.active = True
            else:
                if (
                    self.active
                    and (self.total_samples - self.last_speech_sample)
                    >= self.min_silence_samples
                ):
                    self.active = False
                    print("End of utterance detected!")
                    utterance = list(self.buffer)[-self.min_silence_samples :]
                    return True, np.array(utterance)

        return False, None
