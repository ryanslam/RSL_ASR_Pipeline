from lib.AudioInputDevices import MicStreamBuilder
from lib.AudioResampler import AudioResampler
from lib.EndOfUtteranceDetector import EOUDetector
from lib.Transcription import Transcriber, TranscriberBuilder
from lib.zmq_publisher import publish_to, close_socket
import torch

TARGET_SR = 16000

def main():
    speech_pub = publish_to(port=5555, bind=True)

    test_buffer = []
    resampler = AudioResampler()
    detector = EOUDetector()
    started_speaking = False
    transcriber = (
                    TranscriberBuilder()
                    .set_device()
                    .set_vad('silero')
                    .set_whisper_model("large-v3")
                    .build()
                    )
    
    def test_callback(indata, frames, time, status):
        nonlocal started_speaking 
        if status:
            print(status)

        # Ensure audio data is mono.
        mono = indata[:, 0] if indata.ndim > 1 else indata

        if resampler:
            mono = resampler.process(mono)

        if detector:
            is_speech = detector.process_block(mono)
            if is_speech:
                started_speaking = True
                test_buffer.append(mono)
            elif started_speaking and not is_speech:
                print('end of utterance detected')
                started_speaking = False
                audio_tensor = torch.cat(test_buffer)
                user_speech = transcriber.transcribe(audio_tensor)
                print(user_speech)
                speech_pub.send_string(user_speech['segments'][-1]['text'])
                if(user_speech['segments']):
                    print(user_speech['segments'][-1]['text'])
                test_buffer.clear()

    input_stream = MicStreamBuilder().set_chunk_size(512).set_dev_idx(10).set_callback(test_callback)
    if isinstance(resampler, AudioResampler):
        input_stream.set_resampler(resampler, TARGET_SR)
    if isinstance(detector, EOUDetector):
        input_stream.set_eou_detector(detector)

    input_stream.build()

    input_stream.start()
    print("Listening... Press Ctrl+C to stop.")
    import time

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        input_stream.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
