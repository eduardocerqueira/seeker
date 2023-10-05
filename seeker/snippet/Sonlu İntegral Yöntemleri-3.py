#date: 2023-10-05T17:09:30Z
#url: https://api.github.com/gists/d9e85b925100fc004a5068d0c53e245e
#owner: https://api.github.com/users/dildeolupbiten

def riemann_integration(f, a, b, n):
    return (h := (b - a) / n) * sum(f(a + i * h) for i in range(n))