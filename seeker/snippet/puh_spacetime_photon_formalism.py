#date: 2025-07-30T16:45:07Z
#url: https://api.github.com/gists/33dd3d6433ae39903327a70f461bdc15
#owner: https://api.github.com/users/BrianMartell


import sympy as sp
from sympy import diff, symbols, Function, sqrt

# Define variables and constants
x, t, phi = symbols('x t phi')  # spacetime coordinate, lattice field
G, c, lambda_c, k, ell_P = symbols('G c lambda k ell_P', positive=True)  # constants
rho_gamma = Function('rho_gamma')(t)  # photon density
g = symbols('g', cls=Function)  # metric determinant (simplified as function)

# Define photon field and field strength (simplified 1D case)
A = Function('A')(x, t)
F = diff(A, x)  # Approximate F_{\mu\nu} as spatial derivative for simplicity

# PUH fold factor
F_fold = k * phi**2 / ell_P**2

# Action density (Lagrangian)
L = - (F**2) / 4 + lambda_c * rho_gamma(t)**2 * F_fold * sqrt(-g(t))

# Approximate Einstein term (simplified R ~ curvature proxy)
R = symbols('R', cls=Function)
L_total = (c**4 / (16 * sp.pi * G)) * R(t) + L

# Vary with respect to metric (g) for stress-energy (simplified)
# dL/dg ~ partial derivative of L with respect to g
dL_dg = diff(L, g(t))  # Stress-energy contribution from photon fold

# Define rho_gamma as a function of time (early universe cooling)
T = symbols('T')  # temperature
rho_gamma_expr = T**4  # Stefan-Boltzmann approximation
dL_dg_sub = dL_dg.subs(rho_gamma(t), rho_gamma_expr)

# Display results
print("Photon Fold Contribution to Stress-Energy (dL/dg):")
sp.pprint(dL_dg_sub)

# Simulate metric evolution (numerical example)
T_0, z = symbols('T_0 z', positive=True)  # initial temperature, redshift
T_expr = T_0 / (1 + z)  # Cooling with expansion
rho_gamma_evol = (T_0 / (1 + z))**4
dL_dg_evol = dL_dg.subs(rho_gamma(t), rho_gamma_evol)

print("\nStress-Energy Evolution with Redshift:")
sp.pprint(dL_dg_evol)

# Solve for phi influence (simplified equilibrium)
eq_phi = diff(F_fold, phi) - lambda_c * rho_gamma_evol * phi  # Equilibrium condition
phi_sol = sp.solve(eq_phi, phi)
print("\nEquilibrium phi Solution:")
sp.pprint(phi_sol)
