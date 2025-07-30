#date: 2025-07-30T17:12:31Z
#url: https://api.github.com/gists/2942dfff79a61eec1fdb16a910b2b179
#owner: https://api.github.com/users/BrianMartell

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables
t, x, y, z = sp.symbols('t x y z')  # Spacetime coordinates
k, E = sp.symbols('k E', positive=True)  # Wavenumber, photon energy
a = sp.Function('a')(t)  # Scale factor
phi = sp.Function('phi')(x, y, z, t)  # Lattice spacetime field
L = sp.Symbol('L', positive=True)  # Lattice spacing (e.g., Planck length)
Delta_E = sp.Symbol('Delta_E', positive=True)  # Energy gap

# Photon energy in lattice (quantized)
n = sp.Symbol('n', integer=True, positive=True)
E_n = n * Delta_E * a**(-2)  # Energy scales with radiation era (a^-2)
photon_energy = E_n.subs(n, sp.Function('n')(x, y, z, t))

# Lattice dynamics (wave equation with photon coupling)
coupling = photon_energy * phi  # Photon-lattice interaction
lattice_eq = sp.diff(phi, t, 2) - (1/a**2) * (sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)) + Delta_E * phi + coupling
solution = sp.dsolve(lattice_eq, phi)

# CMB power spectrum (simplified PUH contribution)
C_ell = sp.Function('C_ell')(k)
interaction_term = sp.exp(-k**2 * L**2) * E_n  # Lattice damping + photon energy
C_ell_expr = sp.integrate(interaction_term**2 / a**2, (k, 0, sp.oo))

# Numerical evaluation
L_val = 1.616e-35  # Planck length (m)
Delta_E_val = 1e-3  # Energy gap (eV)
a_val = 1e-3  # Scale factor at recombination
k_vals = np.logspace(-3, 2, 100)  # Wavenumber range (Mpc^-1)
C_ell_func = sp.lambdify(k, C_ell_expr.subs({L: L_val, Delta_E: Delta_E_val, a: a_val}), 'numpy')
C_ell_vals = [C_ell_func(k) for k in k_vals]

# Plot PUH CMB contribution
plt.figure(figsize=(8, 6))
plt.loglog(k_vals, C_ell_vals, label='PUH Lattice Contribution')
plt.xlabel('Wavenumber k (Mpc^-1)')
plt.ylabel('C_ell (Arbitrary Units)')
plt.title('PUH CMB Power Spectrum Contribution')
plt.legend()
plt.grid(True)
plt.show()

# Evolutionary analysis: lattice field over time
t_vals = np.linspace(1e-4, 1e-3, 100)  # Early universe time range
a_vals = t_vals**2  # Radiation-dominated expansion (a ~ t^2)
phi_evolution = []
for t_val, a_val in zip(t_vals, a_vals):
    phi_t = solution.rhs.subs({t: t_val, a: a_val, L: L_val, Delta_E: Delta_E_val})
    phi_evolution.append(float(phi_t.evalf()))

# Plot lattice evolution
plt.figure(figsize=(8, 6))
plt.plot(t_vals, phi_evolution, label='Lattice Field Evolution')
plt.xlabel('Time (Arbitrary Units)')
plt.ylabel('phi (Arbitrary Units)')
plt.title('PUH Spacetime Lattice Evolution')
plt.legend()
plt.grid(True)
plt.show()

# CMB comparison (placeholder for Planck data)
# Example: chi^2 fit (requires actual Planck C_ell data)
# planck_C_ell = load_planck_data()  # Pseudo-code
# chi2 = sum((C_ell_vals - planck_C_ell)**2 / planck_errors**2)