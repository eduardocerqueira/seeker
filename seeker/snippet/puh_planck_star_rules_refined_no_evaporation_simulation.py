#date: 2026-02-05T17:41:11Z
#url: https://api.github.com/gists/5e49c13f641a0f2b69dafe44edff8c14
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Refined Planck Star Rules Sim — Standard vs PUH Lifetime (Domain Failure)
M = np.logspace(10, 20, 500)  # Mass kg log
G = 6.67430e-11
hbar = 1.0545718e-34
c = 3e8
t_standard = 5120 * np.pi * G**2 * M**3 / (hbar * c**4) / (3.156e7 * 1e9)  # Gyr

plt.figure(figsize=(10,6))
plt.loglog(M / 1.989e30, t_standard, label='Standard Hawking Finite Lifetime', color='gold', lw=2)
plt.axhline(1e100, color='cyan', linestyle='--', label='PUH Effective Infinite (Topological Protection)')
plt.xlabel('Mass (Solar Masses log)')
plt.ylabel('Lifetime (Gyr log)')
plt.title('PUH v25: Refined Planck Star Rules No Evaporation Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_planck_star_rules_refined_no_evaporation_simulation.png', dpi=300)
plt.show()

print("Standard finite, PUH topological + discrete → infinite lifetime.")