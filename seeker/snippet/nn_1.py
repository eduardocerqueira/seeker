#date: 2025-01-01T16:34:49Z
#url: https://api.github.com/gists/bf5c4b749a6432160cc4c6e543ff0b0e
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import matplotlib.pyplot as plt

# Generate a simple sine wave
time_steps = np.linspace(-np.pi*2, np.pi*2, 1000)
sine_wave = np.sin(time_steps)

# Prepare the dataset
X, y = [], []
t_X, t_Y = [],[]
window_size = 20

for i in range(len(sine_wave) - window_size):
    X.append(sine_wave[i:i + window_size])
    y.append(sine_wave[i + window_size])
    t_X.append(time_steps[i:i+window_size])
    t_Y.append(time_steps[i+window_size])

X = np.array(X)
y = np.array(y)

# Reshape X for the neural network (samples, time steps)
X = X.reshape((X.shape[0], X.shape[1]))

# Plot the sine wave
plt.plot(time_steps, sine_wave)
plt.title("Sine Wave")
plt.show()