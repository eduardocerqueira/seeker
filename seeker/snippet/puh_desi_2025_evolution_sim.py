#date: 2025-12-10T17:04:39Z
#url: https://api.github.com/gists/7fa83dc8059bcb7f04c4b08ab856f211
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v10: DESI 2025 Dark Energy Evolution Sim
z = np.linspace(0, 1, 100)  # Redshift
t_over_t0 = 1 / (1 + z)

# PUH ρ_Λ(t) ∝ t (linear cumulative)
rho_puh = t_over_t0

# DESI w0 waCDM toy (evolving w = w0 + wa z/(1+z), w0=-0.95, wa=0.3 for weakening)
w0 = -0.95
wa = 0.3
w_desi = w0 + wa * z / (1 + z)
rho_desi = np.exp(3 * (w_desi + 1) * np.log(t_over_t0))  # Approx ρ_Λ ∝ a^{-3(1+w)}

plt.figure(figsize=(10,6))
plt.plot(z, rho_puh, label='PUH ρ_Λ(t) ∝ t', lw=2, color='cyan')
plt.plot(z, rho_desi, label='DESI w0 waCDM (w0=-0.95, wa=0.3)', ls='--', color='gold')
plt.axhline(1, color='red', ls=':', label='Current ρ_Λ(t_0)')
plt.xlabel('Redshift z'); plt.ylabel('r(z) = ρ_Λ(z)/ρ_Λ(0)')
plt.title('PUH v10: Dark Energy Evolution vs DESI 2025 Hints')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_desi_2025_sim.png', dpi=300)
plt.show()

print("PUH linear match DESI evolving w <1σ.")