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

for mel_file in glob('tetiana_samples/*.mel'):
    ts_start = time.time()

    wav_file = mel_file.replace('.mel', '.wav')

    mel_1 = torch.load(mel_file).to('cpu')
    audio = model.decode(mel_1)

    ts_end = time.time() -  ts_start
    print('Inference time: ', ts_end)

    torchaudio.save(wav_file, audio.cpu(), 48000, compression=128)