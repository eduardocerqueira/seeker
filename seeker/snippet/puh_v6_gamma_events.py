#date: 2025-10-10T17:06:53Z
#url: https://api.github.com/gists/1fe9c45532b075d5458cfc2eaaec3d5e
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Constants
time = np.linspace(0, 1, 100)  # Time (s)
frequency = 50  # Gamma band (Hz)
phi_e8 = 1e26  # E8 fold factor
arousal = 1.5  # Arousal multiplier

# Simulate gamma event rate and coherence
event_rate = 0.1 * np.sin(2 * np.pi * frequency * time) * arousal * phi_e8 / 1e26
coherence = np.exp(-0.5 * (time - 0.5)**2 / 0.1**2) * event_rate  # Gaussian bouts

# Plot
plt.figure(figsize=(6, 5))
plt.plot(time * 1000, event_rate, 'b-', label='Event Rate')
plt.plot(time * 1000, coherence, 'r-', label='Neural Coherence')
plt.xlabel('Time (ms)')
plt.ylabel('Value')
plt.title('PUH v6: Gamma Events as E8 Lattice Synchronization')
plt.legend()
plt.grid(True)
plt.savefig('gamma_events.png')
plt.show()

print(f"Average Event Rate: {np.mean(event_rate):.2f}")