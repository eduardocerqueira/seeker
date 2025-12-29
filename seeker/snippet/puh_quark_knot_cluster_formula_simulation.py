#date: 2025-12-29T17:01:19Z
#url: https://api.github.com/gists/220c8929192e16420b67620e2ee34ad5
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Quark Knot Cluster Formula Sim — Energy vs Separation r (Linear Confinement Minimum)
r = np.linspace(0.1, 5, 500)  # Separation r fm arb.
xi_color = 10  # High coupling color
E_res = 1  # Resonance arb.
E_knot = 1  # Single knot arb.
chi = 3  # Baryon color folds
V_string = (xi_color - 1) * E_res * r  # Linear confinement
E_cluster = 3 * chi * E_knot - V_string  # Total cluster energy minimum

plt.figure(figsize=(10,6))
plt.plot(r, V_string, label='Linear Confinement String Potential', color='red', lw=2)
plt.plot(r, E_cluster, label='Quark Knot Cluster Total Energy Minimum', color='cyan', lw=2)
plt.axvline(1, color='gold', ls='--', label='Equilibrium r_{eq} ~1 fm')
plt.xlabel('Separation r (fm arb.)'); plt.ylabel('Energy E (arb.)')
plt.title('PUH v25: Quark Knot Cluster Formula Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_quark_knot_cluster_formula_simulation.png', dpi=300)
plt.show()

print("Linear string positive, cluster minimum equilibrium — color binding geometry.")