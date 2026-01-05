#date: 2026-01-05T17:00:43Z
#url: https://api.github.com/gists/6a3a7965bf00359bcf2e661362c82b39
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Golden ratio
phi = (1 + np.sqrt(5)) / 2

# Lifecycle stages
stages = ['Photon', 'Particle', 'Black Hole', 'Evaporation']
stage_num = np.arange(4)

# Tension scaling by φ^2
tension = [0.01] + [phi**(2*i) for i in range(1,3)] + [phi**4 / 10]  # Peak then relax

plt.figure(figsize=(10,6))
plt.semilogy(stage_num, tension, 'o-', lw=3, markersize=10, color='gold')
for i, txt in enumerate(stages):
    plt.annotate(txt, (stage_num[i], tension[i]), xytext=(5,5), textcoords='offset points', fontsize=12)

plt.xlabel('Lifecycle Stage')
plt.ylabel('Relative Tension (log)')
plt.title('PUH Lifecycle: φ Tension Scaling')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"φ scaling factor φ² ≈ {phi**2:.3f}")