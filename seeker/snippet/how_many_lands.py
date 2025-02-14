#date: 2025-02-14T17:03:27Z
#url: https://api.github.com/gists/5525baab8c77f6f8dc504a657344adf7
#owner: https://api.github.com/users/Spigushe

import math
import pandas as pd

DRAW = False
SIZE = 99
LANDS = [10 + i for i in range(40 + 1)]
COSTS = [
    "C",
    "1C",
    "CC",
    "2C",
    "1CC",
    "CCC",
    "3C",
    "2CC",
    "1CCC",
    "4C",
    "3CC",
    "2CCC",
    "5C",
    "4CC",
    "3CCC",
]


def nCr(n, r):
    """Number of cominations

    Args:
        n (int): Total number of items
        r (int): Number of items to choose

    Returns:
        int: Number of combinations
    """
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def hnpss(n: int, p: int, s: int, e: int):
    """Hypergeomtric probabilities of having exactly `e` success

    Args:
        n (int): number of draws
        p (int): population size
        s (int): number of success states in the population
        e (int): number of observed successes

    Returns:
        prob (float): probability of having exactly `e` success items
    """
    return nCr(s, e) * nCr(p - s, n - e) / nCr(p, n)


def cost_to_int(cost: str) -> int:
    """Read cost from string (ex: 1CC returns 3)

    Args:
        cost (str): cost string

    Returns:
        int: cost
    """
    total = cost.count("C")
    if len(cost) == total:
        return total
    return int(cost[:1]) + total


results = []

for land in LANDS:
    line = []
    for cost in COSTS:
        total = 0
        for i in range(cost.count("C")):
            total += hnpss(6 + DRAW + 1 * (cost_to_int(cost)), SIZE, land, i)
        line.append(1 - total)
    results.append(line)

df = pd.DataFrame(results, index=LANDS, columns=COSTS)
print(df)
