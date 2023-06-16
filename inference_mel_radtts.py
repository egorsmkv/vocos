import time

import torch
import torchaudio
from glob import glob

from vocos import Vocos

model = Vocos.from_hparams("my_config.yaml")

raw_model = torch.load('/home/yehor/Work/github/vocos/logs/lightning_logs/version_1/checkpoints/vocos_checkpoint_epoch=336_step=214332_val_loss=10.2031.ckpt', map_location="cuda")

model.load_state_dict(raw_model['state_dict'], strict=False)
model.eval()

model.to('cuda')

for mel_file in glob('tetiana_samples/*.mel'):
    ts_start = time.time()

    wav_file = mel_file.replace('.mel', '.wav')

    mel_1 = torch.load(mel_file)
    audio = model.decode(mel_1)

    ts_end = time.time() -  ts_start
    print('Inference time: ', ts_end)

    torchaudio.save(wav_file, audio.cpu(), 22050)
