#date: 2026-01-05T17:09:07Z
#url: https://api.github.com/gists/bd42fb1aa796c4aeea02ad8e64215993
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Golden ratio
phi = (1 + np.sqrt(5)) / 2

# Pentagonal angles (degrees)
angles_deg = np.array([0, 72, 144, 216, 288])
angles_rad = np.deg2rad(angles_deg)

# Unit circle points
x = np.cos(angles_rad)
y = np.sin(angles_rad)

plt.figure(figsize=(8,8))
plt.scatter(x, y, s=150, color='gold', zorder=5)
for i, ang in enumerate(angles_deg):
    plt.annotate(f'{ang}°\ncos={np.cos(np.deg2rad(ang)):.4f}', (x[i], y[i]), xytext=(10,10), textcoords='offset points', fontsize=10)
plt.plot(np.cos(np.linspace(0,2*np.pi,200)), np.sin(np.linspace(0,2*np.pi,200)), 'k--', alpha=0.5)
plt.axis('equal')
plt.title('PUH E₈ Projection: Pentagonal Symmetry Yielding φ')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"cos(72°) = {(phi - 1)/2:.6f}")
print("φ from E₈ 5-fold rotational symmetry.")