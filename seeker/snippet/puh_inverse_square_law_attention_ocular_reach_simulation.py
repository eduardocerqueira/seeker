#date: 2026-01-22T17:12:56Z
#url: https://api.github.com/gists/89c1336320af8b5b289b320d7410073b
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Inverse Square Law Attention Ocular Reach Sim — \Delta \xi vs Distance (1/r^2 Drop)
r = np.logspace(-2, 2, 500)  # Distance m log 1 cm to 100 m
gain = 1e18  # Phased-array max toy
delta_xi = gain / r**2  # Stiffening \propto 1/r^2

plt.figure(figsize=(10,6))
plt.loglog(r, delta_xi, label='Attention Stiffening \Delta \xi \propto 1/r^2', color='cyan', lw=2)
plt.axvline(1, color='gold', ls='--', label='1 Meter Distance')
plt.xlabel('Distance r (m log)'); plt.ylabel('\Delta \xi Stiffening (arb. log)')
plt.title('PUH v25: Inverse Square Law Attention Ocular Reach Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_inverse_square_law_attention_ocular_reach_simulation.png', dpi=300)
plt.show()

print("Maximum at eye, 1/r^2 drop, measurable at 1 m — gaze power falloff.")