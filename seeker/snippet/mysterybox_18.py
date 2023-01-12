#date: 2023-01-12T17:18:29Z
#url: https://api.github.com/gists/9169787322dc33f0bb48fc2e440583a1
#owner: https://api.github.com/users/dennisbakhuis

import haversine
from scipy.optimize import minimize

def func(a, b, c, d, x1, y1, A):
    d1 = haversine.haversine((a, b), (x1, y1), unit='m')
    d2 = haversine.haversine((c, d), (x1, y1), unit='m')
    return abs(d1 - d2 - A)

def calc_error(X):
    a, b, c, d = X[0], X[1], X[2], X[3]
    error = 0
    for row in gps.itertuples():
        A = row.A
        x1 = row.latitude
        y1 = row.longitude
        error += func(a, b, c, d, x1, y1, A)
    return error / len(gps)


bounds = [
    (45, 60), (-10, 16),
    (45, 60), (-10, 16),
]

# Some random guesses in Enschede  
X0 = [  
    52.214440, 6.881985,
    52.223679, 6.907095,
]

# Minimize to find p1 and p2.
res = minimize(
    calc_error,
    x0=X0,
    bounds=bounds,
    tol=1e-4,
)

x = res.x
p1, p2 = (x[0], x[1]), (x[2], x[3])
goal = (x[0] + x[2]) /2, (x[1] + x[3]) / 2
print(f"Goal location: {goal}") 