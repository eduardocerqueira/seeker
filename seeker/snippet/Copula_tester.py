#date: 2025-03-14T17:04:24Z
#url: https://api.github.com/gists/89594cd2c78f2aba98afb19ee698f31d
#owner: https://api.github.com/users/Clement1nes

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copulae import GaussianCopula, StudentCopula, ClaytonCopula, GumbelCopula, FrankCopula


def pseudo_observations(data: pd.DataFrame) -> np.ndarray:
    """Convert price data to pseudo-observations in [0, 1]."""
    ranks = data.rank(method="average")
    n = len(data)
    return ((ranks - 0.5) / n).values  # Convert to NumPy array


def plot_copula_fit(u: np.ndarray, best_copula_name: str, best_copula):
    """Visualizes the fitted copula by sampling from it."""
    try:
        simulated_data = best_copula.random(len(u))  # Generate synthetic observations
    except Exception as e:
        print(f"Error generating simulated data for {best_copula_name}: {e}")
        return

    # Convert to NumPy array for safe indexing
    u = np.array(u)
    simulated_data = np.array(simulated_data)

    if simulated_data.shape[1] != 2:  # Ensure it's a 2D copula
        print(f"Unexpected shape of simulated data: {simulated_data.shape}")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(u[:, 0], u[:, 1], alpha=0.5, label="Actual Data", color="blue")
    plt.scatter(simulated_data[:, 0], simulated_data[:, 1], alpha=0.5, label="Simulated Data", color="red")
    plt.xlabel("Asset 1 Pseudo-Observations")
    plt.ylabel("Asset 2 Pseudo-Observations")
    plt.title(f"Copula Fit: {best_copula_name}")
    plt.legend()
    plt.show()


def main():
    csv_file_path = "MASKVET22-23.csv"

    # Read CSV and determine header dynamically
    df = pd.read_csv(csv_file_path, header=None)  # No assumption about headers

    # Print first rows to check structure
    print("Raw Data Preview:")
    print(df.head(10))

    # Manually rename columns if needed
    df.columns = ['timestamp', 'close_1', 'close_2', 'volume_1', 'volume_2', 'spread_st']

    # Print column names to verify correct parsing
    print("Fixed Columns:", df.columns.tolist())

    # Ensure necessary columns exist
    required_columns = ['close_1', 'close_2']
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")

    # Drop NaNs in price columns
    df.dropna(subset=required_columns, inplace=True)

    # Convert price data to pseudo-observations
    data = df[required_columns].copy()
    u = pseudo_observations(data)

    # Fit different copulas
    copulas = {
        "Gaussian": GaussianCopula(dim=2),
        "Student": StudentCopula(dim=2),
        "Clayton": ClaytonCopula(),
        "Gumbel": GumbelCopula(),
        "Frank": FrankCopula()
    }

    results = []
    for name, cop in copulas.items():
        try:
            cop.fit(u, method='ml')
            log_lik = cop.log_lik(u)
            aic = 2 * len(cop.params) - 2 * log_lik
            results.append((name, log_lik, aic, cop))
        except Exception as ex:
            print(f"Error fitting {name}: {ex}")

    if not results:
        print("No copulas successfully fitted.")
        return

    results.sort(key=lambda x: x[2])  # Sort by AIC (lower is better)
    best_copula_name, _, _, best_copula = results[0]

    # Print results
    print("=== Copula Fit Results ===")
    for cop_name, ll, aic, _ in results:
        print(f"{cop_name:8} | Log-likelihood: {ll:.4f} | AIC: {aic:.4f}")

    print("\nBest copula:", best_copula_name)

    # Plot copula fit
    plot_copula_fit(u, best_copula_name, best_copula)


if __name__ == "__main__":
    main()