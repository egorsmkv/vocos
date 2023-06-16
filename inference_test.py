import torch
import torchaudio

from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

y, sr = torchaudio.load('example_lada.wav')
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
y_hat = vocos(y)

print(y_hat)
print(y_hat.shape)

torchaudio.save("example_lada_vocos.wav", y_hat, 24000, compression=128)
