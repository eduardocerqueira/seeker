#date: 2025-12-04T16:51:12Z
#url: https://api.github.com/gists/65da9bbf38bcce50b96ac8c22b3998e5
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Amalgamation: number of Planck stars over time
t = np.logspace(0, 15, 1000)  # years
N0 = 1e12  # initial number
t_merge = 1e12  # characteristic timescale
N = N0 * np.exp(-t / t_merge)

plt.figure(figsize=(8,5))
plt.loglog(t, N, 'b-')
plt.xlabel('Time (years)')
plt.ylabel('Number of Planck stars')
plt.title('PUH v6: Planck-Star Amalgamation')
plt.grid(True)
plt.savefig('puh_amalgamation.png', dpi=300)
plt.show()