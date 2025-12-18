#date: 2025-12-18T17:01:12Z
#url: https://api.github.com/gists/fe31eb3f3b33a4453c63c2ae856466dd
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v15: Unclaimed Seeds DM Number Sim — Census
M_vis_obs = 1e53  # kg observable baryons
ratio_dm_b = 5
M_dm_obs = ratio_dm_b * M_vis_obs

# Seed masses toy range (atomic to baseball)
M_seed = np.logspace(10, 25, 100)  # kg
N_seeds = M_dm_obs / M_seed

# Milky Way halo toy
M_dm_gal = 2e42  # kg ~10^12 M_⊙
N_gal = M_dm_gal / M_seed

plt.figure(figsize=(10,6))
plt.loglog(M_seed, N_seeds, label='Observable Universe Seeds', color='cyan', lw=2)
plt.loglog(M_seed, N_gal, label='Per Galaxy Halo Seeds', color='gold', lw=2)
plt.axvline(2e25, color='red', ls='--', label='Baseball Seed ~2e25 kg')
plt.xlabel('Seed Mass M (kg)'); plt.ylabel('Number N')
plt.title('PUH v15: Unclaimed Seeds DM Census Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_dark_matter_unclaimed_seeds_number_simulation.png', dpi=300)
plt.show()

print(f"Baseball: Observable N ~{M_dm_obs / 2e25:.1e}")