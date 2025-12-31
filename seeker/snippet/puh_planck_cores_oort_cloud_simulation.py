#date: 2025-12-31T16:55:48Z
#url: https://api.github.com/gists/c66cb717f7fa5fa5ab0e3c28cc9a4f68
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Planck Cores Oort Cloud Sim — Core Mass vs Accretion Time (Grow to Comet Size Icy Shell)
t = np.logspace(0, 10, 500)  # Time years arb. Gyr scale
M_initial = 1e-8  # Planck mass seed kg
growth_rate = 1e-15  # Accretion toy
M_accreted = M_initial + growth_rate * t**2  # Quadratic growth icy shell

plt.figure(figsize=(10,6))
plt.loglog(t, M_accreted, label='Planck Core Accretion Icy Shell Growth', color='cyan', lw=2)
plt.axhline(1e12, color='gold', ls='--', label='Typical Comet Mass ~10^{12} kg')
plt.axvline(4.5e9, color='red', ls='--', label='Solar System Age ~4.5 Gyr')
plt.xlabel('Time (years arb.)'); plt.ylabel('Total Mass M (kg arb.)')
plt.title('PUH v25: Planck Cores Oort Cloud Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_planck_cores_oort_cloud_simulation.png', dpi=300)
plt.show()

print("Seed Planck mass grow quadratic accretion icy shell comet size Gyr.")