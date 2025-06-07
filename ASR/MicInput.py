import pyaudio
import torch

class MicInput:

    def __init__(self,
                format=pyaudio.paInt16,
                channels=1,
                sample_rate=16000,
                chunk_size=512):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.format = format
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def _get_usb_mic(self):
        device_count = self.audio.get_device_count()

        for device_index in range(device_count):
            device_info = self.audio.get_device_info_by_index(device_index)
            print(device_info)
            if device_info['maxInputChannels'] > 0:
                device_name = device_info['name']
                self.sample_rate = int(device_info['defaultSampleRate'])
                if 'USB' in device_name:
                    return (device_index, device_info)
        
        return None

    def initialize_audio_stream(self) -> None:
        if(mic_index:=self._get_usb_mic()):
            print(f"\nMicrophone Name: {mic_index[1]['name']}")
            print(f"Max Input Channels {mic_index[1]['maxInputChannels']}")
            print(f"Default Sampling Rate {mic_index[1]['defaultSampleRate']}\n")
            self.stream = self.audio.open(format=self.format,
                                            channels=self.channels,
                                            rate=self.sample_rate,
                                            input=True,
                                            input_device_index=mic_index[0],
                                            frames_per_buffer=self.chunk_size)
        else:
            print("Unable to detect mic to begin audio stream!")
            self.stop_listening()
    
    def listen(self) -> None|bytes:
        if not self.stream:
            print("No active stream to begin listening!")
            return
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def stop_listening(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()