import torch
import numpy as np

class VoiceActivityDetector:

    def __init__(self, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.hub.download_url_to_file(kwargs['url'])

        model, utils = torch.hub.load(repo_or_dir=kwargs['repo'],
                                      model=kwargs['model'],
                                      force_reload=True)

        (self.get_speech_timestamps, _, self.read_audio, _) = utils
        self.model = model.to(self.device)