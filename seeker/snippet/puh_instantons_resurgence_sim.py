#date: 2025-12-10T16:55:16Z
#url: https://api.github.com/gists/1098f4e9a750dc2704f330335aff33dd
#owner: https://api.github.com/users/BrianMartell

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# PUH v10: Instantons Resurgence Sim — QFT Borel Plane
S_inst = 8 * np.pi**2  # From E8 roots
g = 0.1  # Coupling
n_terms = 5  # Loops

# Toy coeffs (asymptotic growth)
c_n = np.array([-7, -102, 306, -9000, 25000])  # β-like

def borel_B(t):
    return sum(c_n[k] * t**k / np.math.factorial(k) for k in range(n_terms))

t = np.linspace(-100, 10, 1000)
B_vals = np.array([borel_B(ti) for ti in t])

# Multi-instanton poles
k = np.arange(1, 6)
poles = -S_inst / k

plt.figure(figsize=(10,6))
plt.plot(t, B_vals, lw=2, color='cyan', label='B(t)')
for p in poles:
    plt.axvline(p, color='red', ls='--', alpha=0.7, label=f'-S_inst/{int(k[i])}' if i==0 else "")
plt.xlabel('t'); plt.ylabel('B(t)'); plt.title('PUH v10: Instantons in QFT Resurgence — Borel Poles')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('puh_instantons_resurgence_sim.png', dpi=300)
plt.show()

# Resummed example
def integrand(ti):
    return np.exp(-ti / g) * borel_B(ti) / g

res, err = quad(integrand, 0, np.inf)
print(f"Resummed value ≈ {res:.4f} ± {err:.4e}")