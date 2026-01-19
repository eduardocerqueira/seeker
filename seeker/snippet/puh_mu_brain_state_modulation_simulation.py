#date: 2026-01-19T17:00:05Z
#url: https://api.github.com/gists/ea6433c255eab09d96996586e5a986a0
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Mu Brain State Modulation Sim — Spectrum Mu Balanced vs Gamma Sharp / Delta Broad
f = np.logspace(4, 7, 500)  # Frequency Hz log kHz-MHz
N = 1e9
f_mu = 10 * np.sqrt(N)  # ~316 kHz balanced
f_gamma = 40 * np.sqrt(N)  # ~1.26 MHz sharp
f_delta = 2 * np.sqrt(N)   # ~63 kHz broad
spectrum_mu = np.exp(-(np.log10(f) - np.log10(f_mu))**2 / 0.8)  # Balanced sensorimotor
spectrum_gamma = np.exp(-(np.log10(f) - np.log10(f_gamma))**2 / 0.2)  # Sharp
spectrum_delta = np.exp(-(np.log10(f) - np.log10(f_delta))**2 / 2)  # Broad

plt.figure(figsize=(10,6))
plt.loglog(f, spectrum_mu, label='Mu Balanced Embodied ~316 kHz', color='purple', lw=2)
plt.loglog(f, spectrum_gamma, '--', label='Gamma Sharp Focused', color='cyan', lw=2)
plt.loglog(f, spectrum_delta, '--', label='Delta Broad Restorative', color='gold', lw=2)
plt.xlabel('Frequency (Hz log)'); plt.ylabel('Resonance Amplitude (arb.)')
plt.title('PUH v25: Mu Brain State Modulation Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_mu_brain_state_modulation_simulation.png', dpi=300)
plt.show()

print("Mu balanced mid-resonance embodied sensorimotor, vs gamma sharp / delta broad — brain-state harmonics.")