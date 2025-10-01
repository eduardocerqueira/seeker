#date: 2025-10-01T17:05:21Z
#url: https://api.github.com/gists/06a2bfc55d58953840def5963e67cb48
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Constants
theta_max = 6.5  # Adjusted to 6.5° to reflect 5.5°-6.5° range (degrees)
n_quasars = 1000  # Number of quasars simulated
center_ra = 0.0  # Right ascension of Milky Way center (degrees)
center_dec = 0.0  # Declination of Milky Way center (degrees)

# Simulate quasar positions with refined conical distribution
# Use a Gaussian spread within theta_max for better jet collimation
theta = np.random.normal(loc=theta_max/2, scale=theta_max/6, size=n_quasars)
theta = np.clip(theta, 0, theta_max)  # Ensure within 0 to 6.5°
phi = np.random.uniform(0, 2 * np.pi, n_quasars)
ra = center_ra + np.degrees(theta * np.cos(phi))
dec = center_dec + np.degrees(theta * np.sin(phi))

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(ra, dec, s=10, alpha=0.5, c='blue', label='Quasar Positions')
plt.axvline(center_ra, color='red', linestyle='--', label='Milky Way Center')
plt.axhline(center_dec, color='red', linestyle='--')
plt.xlabel('Right Ascension (degrees)')
plt.ylabel('Declination (degrees)')
plt.title('PUH v6: Quasar Dipole as 6° Conical Jet Imprint (Refined)')
plt.legend()
plt.grid(True)
plt.savefig('quasar_dipole.png')
plt.show()

# Print angular spread
mean_theta = np.mean(np.degrees(theta))
print(f"Mean Angular Spread: {mean_theta:.1f} degrees (Range: 0° to {theta_max:.1f}°)")