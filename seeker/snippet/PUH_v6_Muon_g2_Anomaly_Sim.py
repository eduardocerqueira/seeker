#date: 2025-11-19T17:06:43Z
#url: https://api.github.com/gists/cc9ab480d6f442eb4f2682604678ab25
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# g-2 Anomaly: Strain scaling with m_l / m_e
m_mu_m_e = 206.768
alpha = 1 / 137.035999
torus_factor = 2.4  # Shape
a_mu = (m_mu_m_e) * (alpha / np.pi)**4 * torus_factor

# Plot
plt.figure(figsize=(10, 6))
plt.bar(['Target Old', 'PUH v6', '2025 New'], [2.51e-9, a_mu, 0], color=['red', 'blue', 'green'])
plt.ylabel('Δa_μ')
plt.title('PUH v6: Muon g-2 Geometric Strain')
plt.savefig('puh_muon_g2_anomaly.png', dpi=300)
plt.show()

print("PUH a_mu:", a_mu)