#date: 2024-11-05T16:56:21Z
#url: https://api.github.com/gists/61c3e9cba39abbd5ba4105de2cbafb7a
#owner: https://api.github.com/users/mfffeng

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.stats import norm

def frank_copula_correlation(alpha):
    def integrand(x):
        return x/(np.exp(x) - 1)
    
    # Calculate D1(-alpha) using numerical integration
    integral, _ = quad(integrand, 0, alpha)
    D1 = (1/alpha) * integral + alpha/2
    
    rho_k = 1 - (4/alpha) * (D1 - 1)
    
    # Return difference from target correlation (0.5)
    return rho_k - 0.5

# Initial guess for alpha
alpha_guess = 5.0

# Solve for alpha
alpha = fsolve(frank_copula_correlation, alpha_guess)[0]

print(f"Alpha = {alpha}")

# Now calculate d2 values with r=0
def calculate_d2(S, E, sigma, T):
    return (np.log(S/E) + (-sigma**2/2)*T)/(sigma*np.sqrt(T))

S = 120
E = 130
T = 0.5
sigma1 = 0.3
sigma2 = 0.5

# Calculate d2 values & u values
d2_1 = calculate_d2(S, E, sigma1, T)
d2_2 = calculate_d2(S, E, sigma2, T)
u1 = norm.cdf(d2_1)
u2 = norm.cdf(d2_2)

# Calculate Frank Copula
def frank_copula(u1, u2, alpha):
    numerator = (np.exp(alpha*u1) - 1) * (np.exp(alpha*u2) - 1)
    denominator = np.exp(alpha) - 1
    return (1/alpha) * np.log(1 + numerator/denominator)

# Calculate final option price
option_price = frank_copula(u1, u2, alpha)

print("Results:")
print(f"d2_1 = {d2_1:.4f}")
print(f"d2_2 = {d2_2:.4f}")
print(f"u1 = {u1:.4f}")
print(f"u2 = {u2:.4f}")
print(f"Option Price = {option_price:.4f}")