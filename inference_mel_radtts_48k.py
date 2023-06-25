import time

import torch
import torchaudio
from glob import glob

from vocos import Vocos

model = Vocos.from_hparams("my_config_48k.yaml")

model_path = '/home/yehor/Work/github/vocos/logs/lightning_logs/version_0/checkpoints/last.ckpt'

raw_model = torch.load(model_path, map_location="cpu")

model.load_state_dict(raw_model['state_dict'], strict=False)
model.eval()

mel = torch.randn(1, 80, 256)  # B, C, T

audio = model.decode(mel)

wav_file = 'test.wav'

torchaudio.save(wav_file, audio.cpu(), 48000, compression=128)
