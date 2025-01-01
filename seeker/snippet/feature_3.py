#date: 2025-01-01T17:05:26Z
#url: https://api.github.com/gists/7dca26de0b02cbedd1c6cf395a20baeb
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import matplotlib.pyplot as plt

# Generate the time-domain signal
x = np.linspace(-8*np.pi, 8*np.pi, 10000)
y = np.sin(x) + 0.4*np.cos(2*x) + 2*np.sin(3.2*x)
y= y -np.mean(y)
# Perform the Fourier Transform
Y = np.fft.fft(y)
# Calculate the frequency bins
frequencies = np.fft.fftfreq(len(x), d=(x[1] - x[0]) / (2*np.pi))
# Normalize the amplitude of the FFT
Y_abs = 2*np.abs(Y) / len(x)
# Zero out very small values to remove noise
Y_abs[Y_abs < 1e-6] = 0
relevant_frequencies = np.where((frequencies>0) & (frequencies<10))
Y_phase = np.angle(Y)[relevant_frequencies]
frequencies = frequencies[relevant_frequencies]
Y_abs = Y_abs[relevant_frequencies]

# Plot the magnitude of the Fourier Transform
plt.figure(figsize=(10, 6))
plt.plot(frequencies, Y_abs)
plt.xlim(0, 10)  # Limit x-axis to focus on relevant frequencies
plt.xticks([3.2,1,2])
plt.title('Fourier Transform of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
