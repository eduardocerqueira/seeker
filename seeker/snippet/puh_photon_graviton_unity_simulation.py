#date: 2025-12-26T16:44:30Z
#url: https://api.github.com/gists/c498826261fd4d5e2e6fd4a103ca1b7c
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Photon-Graviton Unity Sim — Single Excitation Surface vs Trapped Modes
x = np.linspace(-10, 10, 500)
excitation = np.exp(-x**2 / 4) * np.cos(5 * x)  # Single ripple wave
photon_surface = excitation  # Free surface propagation
graviton_trapped = excitation * np.exp(-np.abs(x)/2)  # Localized trapped decay

plt.figure(figsize=(10,6))
plt.plot(x, photon_surface, label='Photon Surface Mode (Free c)', color='cyan', lw=2)
plt.plot(x, graviton_trapped, label='Graviton Submerged Trapped (Mass/Inertia)', color='gold', lw=2)
plt.xlabel('Lattice Position x (arb.)'); plt.ylabel('Excitation Amplitude')
plt.title('PUH v25: Photon-Graviton Unity Sim — Same Ripple')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_photon_graviton_unity_simulation.png', dpi=300)
plt.show()

print("Single excitation — surface free photon, trapped graviton knot.")