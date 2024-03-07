#date: 2024-03-07T17:03:28Z
#url: https://api.github.com/gists/e13007693ff884e6d39d15773ae7d7fc
#owner: https://api.github.com/users/JairParra

import numpy as np
from scipy.stats import norm

# Parameters for the call options
S = 49  # Initial stock price
K = 50  # Strike price
r = 0.05  # Risk-free rate
T = 0.3846  # Time to maturity for the first option
T1 = 0.5  # Time to maturity for the second option (Gamma hedge)
sigma = 0.2  # Volatility
delta_S = 1  # Change in stock price

# Black-Scholes formula for call option price and Greeks
def call_option_price(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def delta_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def gamma_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Initial values
C_initial = call_option_price(S, K, r, T, sigma)
Delta_initial = delta_call(S, K, r, T, sigma)
Gamma_initial = gamma_call(S, K, r, T, sigma)
C1_initial = call_option_price(S, K, r, T1, sigma)
Gamma1_initial = gamma_call(S, K, r, T1, sigma)

# Hedging weights
w_T1 = Gamma_initial / Gamma1_initial

# Portfolio value without hedge
S_new = S + delta_S
C_final = call_option_price(S_new, K, r, T, sigma)
portfolio_value_without_hedge = -(C_final - C_initial)

# Portfolio value with Delta-Gamma hedge
C1_final = call_option_price(S_new, K, r, T1, sigma)
Delta1_final = delta_call(S_new, K, r, T1, sigma)
portfolio_value_with_hedge = -(C_final - C_initial) + Delta_initial * delta_S + w_T1 * (C1_final - C1_initial) - Delta1_final * w_T1 * delta_S

print("Initial values:")
print(f"  Call option price (C): {C_initial}")
print(f"  Delta (Δ): {Delta_initial}")
print(f"  Gamma (Γ): {Gamma_initial}")
print(f"  Call option price for hedge (C1): {C1_initial}")
print(f"  Gamma for hedge (Γ1): {Gamma1_initial}")
print(f"  Hedging weight (w_T1): {w_T1}\n")

print(f"Change in stock price: {delta_S}")
print(f"Portfolio value without hedge: {portfolio_value_without_hedge}")
print(f"Portfolio value with Delta-Gamma hedge: {portfolio_value_with_hedge}")
