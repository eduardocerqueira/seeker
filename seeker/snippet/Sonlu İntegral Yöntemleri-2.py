#date: 2023-10-05T17:08:38Z
#url: https://api.github.com/gists/ca48a76c85d0aff702080f5aa2e01452
#owner: https://api.github.com/users/dildeolupbiten

def adaptive_monte_carlo__integration_v2(f, a, b, t):
    def recursive_monte_carlo(f, a, b, t):
        c = (2 * a + b) / 3
        d = (2 * b + a) / 3
        left = (c - a) * f(uniform(a, c))
        mid = (d - c) * f(uniform(c, d))
        right = (b - d) * f(uniform(d, b))
        area = left + mid + right
        if abs(area - (b - a) * f(uniform(a, b))) <= t:
            return area
        else:
            return recursive_monte_carlo(f, a, c, t) + recursive_monte_carlo(f, c, d, t) + recursive_monte_carlo(f, d, b, t)
    return recursive_monte_carlo(f, a, b, t)