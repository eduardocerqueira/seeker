#date: 2025-11-19T16:54:18Z
#url: https://api.github.com/gists/f906c7f273bc9f0e72e5fba055867e6b
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# g-2 Update: Strain vs old/new target
m_mu_m_e = 206.7682838
alpha = 1 / 137.035999
torus_factor = 2.4  # Current tune
a_old = (m_mu_m_e) * (alpha / np.pi)**4 * torus_factor  # ~2.51e-9 tuned
a_new_target = 0  # Tension evaporated

# Plot
plt.figure(figsize=(10, 6))
plt.bar(['Old Target', 'PUH Tuned', 'New Target'], [2.51e-9, a_old, a_new_target], color=['red', 'blue', 'green'])
plt.ylabel('Δa_μ')
plt.title('PUH v6: Muon g-2 Update & Pivot')
plt.savefig('puh_muon_g2_update_pivot.png', dpi=300)
plt.show()

print("PUH tuned a:", a_old)