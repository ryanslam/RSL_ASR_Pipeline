import torch
import numpy as np
from MicInput import MicInput
import cv2
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 _, read_audio,
 *_) = utils
voice = MicInput()
voice.initialize_audio_stream()
data = voice.listen()

def buffer_to_tensor(self, audio_buffer):
    


# while True:

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         print("Now exiting loop...")
#         break
# voice.stop_listening()
# torch.set_num_threads(1)

# from IPython.display import Audio
# from pprint import pprint

# torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True)

# (get_speech_timestamps,
#  _, read_audio,
#  *_) = utils

# sampling_rate = 16000 # also accepts 8000
# wav = read_audio('en_example.wav', sampling_rate=sampling_rate)

# wav = torch.tensor(wav).to(device)
# # get speech timestamps from full audio file
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
# pprint(speech_timestamps)