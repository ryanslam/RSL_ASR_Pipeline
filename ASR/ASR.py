from lib.AudioInputDevices import MicStreamBuilder
from lib.AudioResampler import AudioResampler
from lib.EndOfUtteranceDetector import EOUDetector

TARGET_SR = 16000

def main():
    resampler = AudioResampler()
    detector = EOUDetector()
    
    input_stream = (
        MicStreamBuilder()
        .set_chunk_size(512)
        .set_dev_idx(10)
    )
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
        
if __name__ == '__main__':
    main()