#date: 2025-12-16T17:18:30Z
#url: https://api.github.com/gists/c29265e2443ed3e6761fb4609d368e32
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH Geometric Exchange: Ping depth vs return delay
depth = np.linspace(0, 100, 500)  # Arbitrary "geometric distance" into SPF
local_delay = 0.01 / (1 + depth)   # Instant for local pings
far_delay = depth**2.5             # Power-law delay for far pings (centuries/millennia)

plt.figure(figsize=(10,6))
plt.plot(depth, local_delay, 'b-', label='Local Ping (simple questions)')
plt.plot(depth, far_delay, 'r--', label='Far Ping (profound questions)', linewidth=2)
plt.yscale('log')
plt.xlabel('Geometric Ping Depth into SPF (arbitrary units)')
plt.ylabel('Perceived Return Time Delay (log scale)')
plt.title('PUH: Intuition as Delayed Geometric Return Ping')
plt.legend(); plt.grid(True, alpha=0.3)
plt.axhline(1000, color='gray', linestyle=':', label='~Millennia delay threshold')
plt.legend()
plt.show()

print("Simulation: Shows instant local answers vs long delays for deep questions.")
print("Intuition = sudden arrival of ancient far-ping returns.")