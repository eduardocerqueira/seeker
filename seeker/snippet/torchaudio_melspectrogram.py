#date: 2025-10-06T16:55:15Z
#url: https://api.github.com/gists/d015f5b33f69fc1db5a1b1343e268fdb
#owner: https://api.github.com/users/Manalelaidouni

import torch, torchaudio
import os,sys
import numpy as np

wave = torch.zeros(1, 768 + 1023, dtype=torch.float32)
wave[0,0] = 1.0                  

spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        center=True,
        power=1)

mel = spec(wave)

print(mel)

np.save("mel_cpu.npy", mel.numpy())  


# once i run the above example in two env 

cu  = np.load("mel_cu.npy") # produced with torch and torchaudio 2.8.0+cpu  
cpu = np.load("mel_cpu.npy") # produced with torch and torchaudio 2.8.0+cu126
diff = np.abs(cu - cpu)

print("max abs diff :", diff.max()) # max abs diff : 4.5776367e-05
print("mean abs diff:", diff.mean()) # mean abs diff: 6.1001094e-07