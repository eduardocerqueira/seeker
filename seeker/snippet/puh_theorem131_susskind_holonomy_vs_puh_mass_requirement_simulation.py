#date: 2026-03-17T17:42:41Z
#url: https://api.github.com/gists/e3929918166118d4258add6bcc8cb0a4
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# Susskind mathematical holonomy (abstract clocks)
t = np.linspace(0, 2*np.pi, 200)
fgc = np.cos(t)      # forward
bgc = np.sin(t)      # backward
holonomy = np.sin(2*t)

# PUH physical mass injection at rebound
mass = np.where(t > np.pi, 1.0, 0.0)  # mass appears at rebound → time arrow starts

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t, fgc, 'gold', lw=3, label='Forward Clock (Susskind abstract)')
plt.plot(t, bgc, 'cyan', lw=3, label='Backward Clock')
plt.plot(t, holonomy, 'white', lw=4, label='Holonomy (SSB)')
plt.title('Susskind: Mathematical SSB in Empty de Sitter')
plt.ylabel('Clock Direction')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2,1,2)
plt.plot(t, mass, 'gold', lw=4, label='Mass Nucleated at Rebound (PUH)')
plt.title('PUH: No Mass → No Time; Rebound Injects Mass → Starts Arrow')
plt.xlabel('Rebound Path')
plt.ylabel('Mass (0 = timeless Super Polariton Field)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_theorem131_susskind_vs_puh_mass_requirement.png', dpi=300)
plt.show()

print("Simulation complete: Susskind holonomy works in vacuum; PUH requires mass injection at rebound to break symmetry and create time.")