#date: 2025-01-01T16:40:24Z
#url: https://api.github.com/gists/9c4ba1f586db9711993b3c10a0293587
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import matplotlib.pyplot as plt
time_steps = np.linspace(-1, 1, 500)
def build_sine_waves(num=1000):
  X = []
  freq_ranges = np.linspace(-5,5,num)
  phases = np.linspace(0,2*np.pi,num)
  amplitudes = np.linspace(0,1,num)
  for _ in range(num):
    rand_freq, rand_phase, rand_amp = np.random.choice(freq_ranges), np.random.choice(phases) , np.random.choice(amplitudes)
    X.append(rand_amp*np.sin(2 * np.pi * rand_freq * time_steps + rand_phase))
  return np.array(X)

def build_square_waves(num=1000):
  X = []
  freq_ranges = np.linspace(-5,5,num)
  phases = np.linspace(0,2*np.pi,num)
  amplitudes = np.linspace(0,1,num)
  for _ in range(num):
    rand_freq, rand_phase, rand_amp = np.random.choice(freq_ranges), np.random.choice(phases) , np.random.choice(amplitudes)
    rand_sine = rand_amp*np.sin(2 * np.pi * rand_freq * time_steps + rand_phase)
    square_wave = np.sign(rand_sine)
    X.append(square_wave)
  return np.array(X)