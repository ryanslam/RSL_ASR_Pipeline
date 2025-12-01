from lib.AudioInputDevices import MicStreamBuilder
from lib.AudioResampler import AudioResampler
from lib.EndOfUtteranceDetector import EOUDetector
from lib.Transcription import TranscriberBuilder

import torch


class ASRInitializer:
    def __init__(
        self,
        whisper_model="large-v3",
        min_silence_sec=1,
        lang_code=None,
        target_sr=16000,
        chunk_size=512,
        dev_idx=10,
    ):
        self.resampler = AudioResampler()
        self.detector = EOUDetector(min_silence_sec=min_silence_sec)
        self.transcriber = (
            TranscriberBuilder()
            .set_device()
            .set_vad("silero")
            .set_whisper_model(whisper_model)
            .build()
        )
        self.lang_code = lang_code
        print(f"lang code bias: {self.lang_code}")

        self.target_sr = target_sr
        self.chunk_size = chunk_size
        self.dev_idx = dev_idx

        self.speech_buffer = []
        self.lookback_buffer = []
        self.lookback_duration_sec = 1.0
        self.lookback_max_samples = int(self.lookback_duration_sec * target_sr)
        self.was_speaking = False
        self.user_speech = None

    def _audio_input_callback(self, indata, frames, time, status):
        if status:
            print(status)

        mono = indata[:, 0] if indata.ndim > 1 else indata

        if self.resampler:
            mono = self.resampler.process(mono)

        self.lookback_buffer.append(mono)
        total_lookback_samples = sum(len(chunk) for chunk in self.lookback_buffer)
        while total_lookback_samples > self.lookback_max_samples:
            removed = self.lookback_buffer.pop(0)
            total_lookback_samples -= len(removed)

        is_speech = self.detector.process_block(mono) if self.detector else True

        if is_speech and not self.was_speaking:
            print(f"Recording speech...")
            if self.lookback_buffer:
                for chunk in self.lookback_buffer[:-1]:
                    self.speech_buffer.append(chunk)

        if is_speech:
            self.speech_buffer.append(mono)
        else:
            if self.speech_buffer:
                audio_tensor = torch.cat(self.speech_buffer)
                self.user_speech = self.transcriber.transcribe(audio_tensor, self.lang_code)["segments"]
                self.speech_buffer.clear()

        self.was_speaking = is_speech

    def build_input_stream(self):
        input_stream = (
            MicStreamBuilder()
            .set_chunk_size(self.chunk_size)
            .set_dev_idx(self.dev_idx)
            .set_callback(self._audio_input_callback)
        )
        if isinstance(self.resampler, AudioResampler):
            input_stream.set_resampler(self.resampler, self.target_sr)
        if isinstance(self.detector, EOUDetector):
            input_stream.set_eou_detector(self.detector)
        input_stream.build()
        return input_stream
