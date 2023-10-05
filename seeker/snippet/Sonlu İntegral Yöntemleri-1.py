#date: 2023-10-05T17:08:03Z
#url: https://api.github.com/gists/1e7ce301a0ef69a86397911f67c22ad2
#owner: https://api.github.com/users/dildeolupbiten

from random import uniform

def adaptive_monte_carlo_integration(f, a, b, t):
    def recursive_monte_carlo(f, a, b, t):
        c = (a + b) / 2
        left = (c - a) * f(uniform(a, c))
        right = (b - c) * f(uniform(c, b))
        area = left + right
        if abs(area - (b - a) * f(uniform(a, b))) <= t:
            return area
        else:
            return recursive_monte_carlo(f, a, c, t) + recursive_monte_carlo(f, c, b, t)
        return recursive_monte_carlo(f, a, b, t)