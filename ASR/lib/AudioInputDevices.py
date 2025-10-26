import sounddevice as sd
import logging

DEFAULT_CHUNK_SIZE = 512


class MicStreamBuilder:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )

        # Core stream variables, will be defaulted if not explicitly set.
        self.chunk_size = None
        self.dev_idx = None
        self.dev_sr = None
        self.speaking = False

        # Optional variables (Useful for VAD).
        self.resampler = None
        self.target_sr = None
        self.eou_detector = None

        self.test_buffer = []

        self._list_input_devices()

    def set_dev_idx(self, device_idx):
        try:
            self.dev_idx = device_idx
            dev_info = sd.query_devices(self.dev_idx)
            self.dev_sr = int(dev_info["default_samplerate"])

            self._list_device_info(self.dev_idx)
        except sd.PortAudioError as e:
            self.dev_idx = None
            logging.error(f"Selected index is not available. {e}.")

        return self

    def set_chunk_size(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size

        return self

    def set_resampler(self, resampler, target_sr):
        """
        Uses resampler built using the abstract Audio class.
        """
        self.resampler = resampler
        self.target_sr = target_sr

        return self

    def set_eou_detector(self, detector):
        """
        Uses end of utterance detector using the EndOfUtteranceDetector class.
        """
        self.eou_detector = detector

        return self

    def set_callback(self, callback):
        self._callback = callback

        return self

    def build(self):
        if not self.dev_idx:
            self.dev_idx, dev_info = self._set_default_mic()
            self.dev_sr = int(dev_info["default_samplerate"])
            logging.warning(
                f"No Device set. Defaulting to microphone at idx: {self.dev_idx}"
            )
            self._list_device_info(self.dev_idx)

        if not self.chunk_size:
            self.chunk_size = DEFAULT_CHUNK_SIZE
            logging.warning(
                f"No chunk size has been set. Defaulting to {self.chunk_size}"
            )

        if self.resampler:
            self.resampler.init_resampler(self.dev_sr, self.target_sr)

        self.stream = sd.InputStream(
            device=self.dev_idx,
            channels=1,
            samplerate=self.dev_sr,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=self._callback,
        )

        return self

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)
            
        logging.info("Received audio block")

        # # Ensure audio data is mono.
        # mono = indata[:, 0] if indata.ndim > 1 else indata

        # if self.resampler:
        #     mono = self.resampler.process(mono)

        # if self.eou_detector:
        #     is_speech = self.eou_detector.process_block(mono)
        #     if is_speech:
        #         self.speaking = True
        #         self.test_buffer.append(mono)
        #     elif self.speaking and not is_speech:
        #         print('end of utterance detected')
        #         self.speaking = False
        #         print(self.test_buffer)
        #         self.test_buffer.clear()
        #         print(self.test_buffer)

    def _list_input_devices(self):
        dev_info = sd.query_devices()

        print("\n=====Available Input Devices=====")
        for idx, info in enumerate(dev_info):
            if info["max_input_channels"] > 0:
                mic_name = info["name"]
                print(f"\t{mic_name} at index: {idx}")
        print()

    def _set_default_mic(self):
        input_devices = []
        for idx, dev_info in enumerate(sd.query_devices()):
            if dev_info["max_input_channels"] > 0:
                input_devices.append((idx, dev_info))

        if not input_devices:
            raise ValueError("No Input Devices Currently Detected.")

        default_microphone = input_devices[0]

        return default_microphone

    def _list_device_info(self, dev_idx):
        dev_info = sd.query_devices(dev_idx)

        print(f"\n=====DEVICE INFO (idx: {self.dev_idx})=====")
        for key, val in dev_info.items():
            print(f"\t{key}: {val}")

        print()

        return None

    def start(self):
        logging.info("Starting Audio Stream")
        self.stream.start()

    def stop(self):
        logging.info("Stopping Audio Stream")
        self.stream.stop()
        self.stream.close()


def main():
    input_stream = MicStreamBuilder().set_chunk_size().set_dev_idx(10).build()
    input_stream.start()
    input_stream.stop()


if __name__ == "__main__":
    main()
