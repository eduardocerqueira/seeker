#date: 2025-12-31T17:14:59Z
#url: https://api.github.com/gists/cb1947a6e3b6e11d57eea2881524fdc5
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Primordial Planck Cores Early SMBH Sim — Mass Growth vs Time (Standard Slow Stellar vs PUH Seeded Fast)
t = np.linspace(0, 500, 500)  # Time Myr early universe arb.
M_seed = 1e6  # Primordial seed solar masses toy
t_sal = 40  # Salpeter time Myr
M_puh = M_seed * np.exp(t / t_sal)  # PUH seeded exponential fast
M_standard = 100 * (t / 500)**2  # Standard slow hierarchical toy

plt.figure(figsize=(10,6))
plt.semilogy(t, M_standard, label='Standard Slow Stellar Mergers', color='red', lw=2)
plt.semilogy(t, M_puh, label='PUH Primordial Seeded Rapid Accretion', color='cyan', lw=2)
plt.axhline(1e9, color='gold', ls='--', label='Observed Early SMBH ~10^9 M_\odot')
plt.axvline(400, color='purple', ls='--', label='z>10 Age ~400 Myr')
plt.xlabel('Time (Myr arb.)'); plt.ylabel('Black Hole Mass M (M_\odot arb.)')
plt.title('PUH v25: Primordial Planck Cores Early SMBH Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_primordial_planck_cores_early_smbh_simulation.png', dpi=300)
plt.show()

print("Standard slow hierarchical, PUH seeded exponential fast — early SMBH feasible.")