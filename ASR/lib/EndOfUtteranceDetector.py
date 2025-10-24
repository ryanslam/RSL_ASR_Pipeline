import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Load Silero VAD model
MODEL, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

REQUIRED_SAMPLE_RATE = 16000
REQUIRED_BLOCK_SIZE = 512

# --- Helper functions ---

def array_to_tensor(audio_array: np.ndarray, sample_rate: int) -> torch.Tensor:
    """
    Convert a float32 numpy array (from sounddevice) to a mono torch tensor.
    """
    audio_tensor = torch.from_numpy(audio_array)

    if audio_tensor.ndim > 1 and audio_tensor.shape[1] > 1:
        audio_tensor = audio_tensor.mean(dim=1)
        
    if sample_rate != REQUIRED_SAMPLE_RATE:
        audio_tensor = resample_audio(audio_tensor, orig_sr=sample_rate, target_sr=REQUIRED_SAMPLE_RATE)
    
    return audio_tensor.squeeze()


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(audio)


def set_valid_blocksize(audio_tensor: torch.Tensor) -> torch.Tensor:
    if audio_tensor.shape[-1] > REQUIRED_BLOCK_SIZE:
        audio_tensor = audio_tensor[:REQUIRED_BLOCK_SIZE]
    elif audio_tensor.shape[-1] < REQUIRED_BLOCK_SIZE:
        padding = torch.zeros(REQUIRED_BLOCK_SIZE - audio_tensor.shape[-1])
        audio_tensor = torch.cat([audio_tensor, padding])
    return audio_tensor


def is_speech(audio_tensor: torch.Tensor) -> list:
    with torch.no_grad():
        return get_speech_timestamps(audio_tensor, MODEL, sampling_rate=REQUIRED_SAMPLE_RATE)


# --- Functional End-of-Utterance detector ---
def update_end_of_utterance(audio: np.ndarray, sample_rate: int,
                            last_speech_time: int, active: bool,
                            min_silence_duration: float = 1.0) -> (bool, int, bool):
    """
    Update function for detecting end-of-utterance.
    
    Returns:
        end_detected: bool
        last_speech_time: int
        active: bool
    """
    audio_tensor = array_to_tensor(audio, sample_rate)
    audio_tensor = set_valid_blocksize(audio_tensor)

    segments = is_speech(audio_tensor)

    if segments:
        last_speech_time = segments[-1]['end']
        active = True
        return False, last_speech_time, active
    else:
        num_samples_since_last_speech = audio_tensor.shape[-1] - last_speech_time
        min_silence_samples = int(min_silence_duration * REQUIRED_SAMPLE_RATE)
        
        if active and num_samples_since_last_speech >= min_silence_samples:
            active = False
            print("end of utterance detected")
            return True, last_speech_time, active

    return False, last_speech_time, active
