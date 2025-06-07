import sounddevice as sd
import numpy as np
import torch
import whisper
import queue
import torchaudio
import time

# --- Config ---
MIC_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30  # in ms
FRAME_SIZE = int(MIC_SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_WINDOW_SIZE = 512  # Required by Silero VAD
SILERO_THRESHOLD = 0.5
MAX_SILENCE = 1.0  # seconds

# --- Load Silero VAD ---
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_ts, _, read_audio, _, _) = utils

# --- Load Whisper ---
whisper_model = whisper.load_model("base")

# --- Buffers ---
audio_queue = queue.Queue()
utterance_buffer = []

# --- Device Listing ---
def list_input_devices():
    print("Available input devices:")
    for idx, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']}")

# --- Audio Callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)
    audio_queue.put(indata.copy())

# --- Main Logic ---
def run_vad(device_index):
    vad_audio_buffer = torch.zeros(0)
    last_speech_time = None

    stream = sd.InputStream(
        samplerate=MIC_SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        device=device_index,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    print("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            block = audio_queue.get()
            block = torch.from_numpy(block.squeeze()).float()

            # Resample block to 16kHz for VAD
            block_resampled = torchaudio.functional.resample(block, MIC_SAMPLE_RATE, TARGET_SAMPLE_RATE)
            vad_audio_buffer = torch.cat((vad_audio_buffer, block_resampled))

            # Run VAD in fixed 512-sample chunks
            while vad_audio_buffer.shape[0] >= VAD_WINDOW_SIZE:
                window = vad_audio_buffer[:VAD_WINDOW_SIZE]
                vad_audio_buffer = vad_audio_buffer[VAD_WINDOW_SIZE:]

                speech_prob = vad_model(window.unsqueeze(0), TARGET_SAMPLE_RATE).item()

                if speech_prob > SILERO_THRESHOLD:
                    utterance_buffer.append(block)
                    last_speech_time = time.time()
                    break  # Don't drain further in this loop

            # Check for end of utterance
            if last_speech_time and (time.time() - last_speech_time) > MAX_SILENCE:
                if utterance_buffer:
                    print("End of utterance detected. Transcribing...")

                    full_audio = torch.cat(utterance_buffer)
                    full_audio_16k = torchaudio.functional.resample(full_audio, MIC_SAMPLE_RATE, 16000).numpy()

                    result = whisper_model.transcribe(full_audio_16k, fp16=False)
                    print("🗣️ Transcription:", result['text'])

                    utterance_buffer.clear()
                    last_speech_time = None

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        stream.close()

# --- Entry Point ---
if __name__ == "__main__":
    list_input_devices()
    device_index = int(input("Select input device index: "))
    run_vad(device_index)
