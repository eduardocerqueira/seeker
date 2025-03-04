#date: 2025-03-04T17:05:19Z
#url: https://api.github.com/gists/0a7614e314ce94ee0c7d6a439034db34
#owner: https://api.github.com/users/amstocker

import numpy as np
import wave

from random import randint


sample_rate = 48000
samples_per_cycle = 256

for z in range(8):
    filename = "{}.wav".format(z + 1)
    with wave.open(filename, mode="wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for y in range(8):
            for x in range(8):
                t = np.linspace(0, 1, samples_per_cycle + 1)
                
                # generate a waveform with 10 random harmonics
                output = 0.0 * t
                for harmonic in [randint(1, 10) for _ in range(10)]:
                    output += np.sin(2 * np.pi * harmonic * t)
                
                # normalize waveform and convert to 16 bit integer
                output /= np.max(np.abs(output))
                output = (output * (2 ** 15 - 1)).astype("<h")
                
                wav_file.writeframes(output[:samples_per_cycle])