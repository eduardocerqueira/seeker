#date: 2026-01-23T17:04:21Z
#url: https://api.github.com/gists/6a72ad923826532a0845c05b3c3cf9e2
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Shell Equilibrium Frequency Formula Sim — Frequency Uptick Chirp (Shell Relaxation)
t = np.linspace(0, 50, 500)  # Time arb. post-merger
f_final = 1  # Normalized final frequency
alpha = 0.4  # Initial deformation ~40%
tau = 15  # Damping time arb.
f_t = f_final * (1 + alpha * np.exp(-t / tau))  # Uptick rise
amplitude = np.exp(-t / 20)  # Overall damping envelope
ringdown = amplitude * np.sin(2 * np.pi * f_t.cumsum() / 50)  # Chirp signal

plt.figure(figsize=(10,6))
plt.plot(t, ringdown, label='Ringdown Uptick Chirp (Shell Equilibrium)', color='cyan', lw=2)
plt.xlabel('Time Post-Merger (arb.)'); plt.ylabel('Amplitude (arb.)')
plt.title('PUH v25: Shell Equilibrium Frequency Formula Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_shell_equilibrium_frequency_formula_simulation.png', dpi=300)
plt.show()

print("Initial low frequency, relaxation uptick to final, damped chirp — shell equilibrium whine.")