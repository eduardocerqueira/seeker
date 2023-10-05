#date: 2023-10-05T17:07:35Z
#url: https://api.github.com/gists/291b399ca21b237fef85720f2607b82a
#owner: https://api.github.com/users/dildeolupbiten

from math import pi
from random import uniform

def monte_carlo_integration(f, a, b, n):
    return (b - a) / n * sum(f(uniform(a, b)) for _ in range(n))