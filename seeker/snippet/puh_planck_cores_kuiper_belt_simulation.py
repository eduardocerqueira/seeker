#date: 2025-12-31T17:01:44Z
#url: https://api.github.com/gists/a9ae007d0acaf9346ebd5cf954fc2a10
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Planck Cores Kuiper Belt Sim — Core Mass vs Accretion Time (Denser Faster Growth vs Oort Sparse)
t = np.logspace(0, 10, 500)  # Time years arb. Gyr scale
M_initial = 1e-8  # Planck seed kg
growth_kuiper = 1e-13  # Denser faster accretion toy
growth_oort = 1e-16  # Sparse slower
M_kuiper = M_initial + growth_kuiper * t**2
M_oort = M_initial + growth_oort * t**2

plt.figure(figsize=(10,6))
plt.loglog(t, M_kuiper, label='Kuiper Belt Denser Faster Growth', color='cyan', lw=2)
plt.loglog(t, M_oort, '--', label='Oort Cloud Sparse Slower Growth', color='gold', lw=2)
plt.axhline(1e18, color='red', ls='--', label='Dwarf Planet Mass ~10^{18} kg Pluto Scale')
plt.axvline(4.5e9, color='purple', ls='--', label='Solar System Age ~4.5 Gyr')
plt.xlabel('Time (years arb.)'); plt.ylabel('Total Mass M (kg arb.)')
plt.title('PUH v25: Planck Cores Kuiper Belt Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_planck_cores_kuiper_belt_simulation.png', dpi=300)
plt.show()

print("Kuiper denser faster accretion dwarf planet scale, Oort sparse slower small.")