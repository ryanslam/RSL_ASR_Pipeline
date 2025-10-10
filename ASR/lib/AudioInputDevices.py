import sounddevice as sd


class AudioListener:
    def __init__(self, stream):
        self.stream = stream

    def listen(self):
        if self.stream:
            self.stream.start()

    def stop_listening(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class InputDevicesBuilder:

    def __init__(self):
        self.input_devices = {}
        self.microphone = None
        self.sample_rate = None
        self.block_size = None
        self.chunk_size = 512
        self.target_sample_rate = None
        self.callback_fn = self.default_callback

    def list_input_devices(self):
        if not self.input_devices:
            self.input_devices = self.get_input_devices()

        for idx, dev in self.input_devices.items():
            print(f'{idx}: {dev["name"]}')

        return self

    def get_input_devices(self):
        input_devices = {}
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                input_devices[dev["index"]] = dev
        return input_devices

    def set_input_device(self, dev_idx):
        if not self.input_devices:
            self.input_devices = self.get_input_devices()
        if dev_idx not in self.input_devices:
            raise ValueError(f"No input device found for index {dev_idx}")
        self.microphone = self.input_devices[dev_idx]
        self.sample_rate = self.microphone["default_samplerate"]

        return self

    def set_callback_function(self, callback_fn):
        self.callback_fn = callback_fn
        return self

    def default_callback(self, indata, frames, time, status):
        raise NotImplementedError(
            "No callback function set. Please provide a callback using set_callback_function()."
        )

    def set_sample_rate(self, sample_rate: int):
        if self.sample_rate != sample_rate:
            self.target_sample_rate = sample_rate

        return self

    def set_chunk_size(self, chunk_size: int):
        self.chunk_size = chunk_size

        return self

    def build(self):
        try:
            if not self.target_sample_rate:
                self.target_sample_rate = int(self.sample_rate)

            self.block_size = int(
                self.chunk_size * (self.sample_rate / self.target_sample_rate)
            )

            stream = AudioListener(
                sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=self.microphone["index"],
                    channels=1,
                    dtype="float32",
                    callback=self.callback_fn,
                )
            )
            return stream
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Device configuration error: {e}")
        except sd.PortAudioError as e:
            raise RuntimeError(f"Audio stream error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unknown error during build: {e}")
