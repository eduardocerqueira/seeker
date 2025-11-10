#date: 2025-11-10T16:59:21Z
#url: https://api.github.com/gists/3f0edb68fd4ec2744bcd85d349d8b71b
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# SN Age Bias: Magnitude vs. Age, redshift-dependent correction
age = np.linspace(0, 10, 1000)  # Gyr
z = np.linspace(0, 2, 1000)  # Redshift
bias = 0.05 * age  # mag/Gyr
magnitude = -19.3 + bias  # Base m_B
corrected = magnitude - bias * (z / 2)  # Mock redshift correction

# Plot
plt.figure(figsize=(10, 6))
plt.plot(age, magnitude, 'b-', label='Observed m (Bias)')
plt.plot(age, corrected, 'r--', label='Corrected m (PUH: Fold Evolution)')
plt.xlabel('Age (Gyr)'); plt.ylabel('Magnitude')
plt.title('PUH v6: SN Progenitor Age Bias Correction')
plt.legend(); plt.grid(True)
plt.savefig('puh_sn_age_bias.png', dpi=300)
plt.show()

print("Bias at age=5 Gyr:", bias[500])
print("Corrected m at age=5:", corrected[500])