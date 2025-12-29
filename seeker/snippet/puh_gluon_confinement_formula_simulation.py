#date: 2025-12-29T16:52:24Z
#url: https://api.github.com/gists/b0f451cb6cb97c3a661abb4952d8436e
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: Gluon Confinement Formula Sim — Potential vs Separation r (Coulomb Free vs Linear Confined)
r = np.linspace(0.1, 5, 500)  # Separation r fm arb.
xi_color = 10  # High coupling color
E_res = 1  # Resonance energy arb.
V_coulomb = -1 / r  # Standard perturbative Coulomb free
V_confinement = (xi_color - 1) * E_res * r  # PUH linear string confined

plt.figure(figsize=(10,6))
plt.plot(r, V_coulomb, label='Standard Perturbative Coulomb Free', color='red', lw=2)
plt.plot(r, V_confinement, label='PUH High-\xi Linear Confinement String', color='cyan', lw=2)
plt.xlabel('Separation r (fm arb.)'); plt.ylabel('Potential V(r) (arb.)')
plt.title('PUH v25: Gluon Confinement Formula Sim')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_gluon_confinement_formula_simulation.png', dpi=300)
plt.show()

print("Coulomb free negative, PUH linear positive confined — high-\xi trapped string.")