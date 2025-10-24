import torch
import numpy as np
from collections import deque
import sounddevice as sd
import torchaudio.transforms as T

# --- Load Silero VAD ---
MODEL, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
(get_speech_timestamps, _, _, _, _) = utils
TARGET_SAMPLE_RATE = 16000

# --- End-of-Utterance Detector ---
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
            segments = get_speech_timestamps(vad_input, MODEL, sampling_rate=TARGET_SAMPLE_RATE)

            if segments:
                self.last_speech_sample = self.total_samples - len(vad_input) + segments[-1]['end']
                self.active = True
            else:
                if self.active and (self.total_samples - self.last_speech_sample) >= self.min_silence_samples:
                    self.active = False
                    print("End of utterance detected!")
                    utterance = list(self.buffer)[-self.min_silence_samples:]
                    return True, np.array(utterance)

        return False, None

# --- Resampler wrapper ---
class ResampleIfNeeded:
    def __init__(self, device_samplerate: int, target_sr: int = TARGET_SAMPLE_RATE):
        self.device_sr = int(device_samplerate)
        self.target_sr = target_sr
        if self.device_sr != self.target_sr:
            self.resampler = T.Resample(orig_freq=self.device_sr, new_freq=self.target_sr)
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

# --- Mic Stream wrapper ---
class MicStream:
    def __init__(self, device_index=None, chunk_size=512, detector: EOUDetector = None):
        self.device_index = device_index
        self.chunk_size = chunk_size
        self.detector = detector

        # get device default samplerate
        dev_info = sd.query_devices(self.device_index)
        self.device_sr = int(dev_info['default_samplerate'])

        # setup resampler
        self.resampler = ResampleIfNeeded(self.device_sr)

        # create input stream
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.device_sr,
            blocksize=self.chunk_size,
            dtype='float32',
            callback=self._callback
        )

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        audio_16k = self.resampler.process(mono)
        if self.detector:
            self.detector.process_block(audio_16k)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

# --- Usage Example ---
if __name__ == "__main__":
    # list devices
    print("Available input devices:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"{idx}: {dev['name']} (Default SR: {dev['default_samplerate']})")
    device_idx = int(input("Select device index: "))

    detector = EOUDetector(min_silence_sec=1.0, vad_window_sec=0.3)
    mic = MicStream(device_index=device_idx, chunk_size=512, detector=detector)

    mic.start()
    print("Listening... Press Ctrl+C to stop.")
    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        mic.stop()
        print("Stopped.")
