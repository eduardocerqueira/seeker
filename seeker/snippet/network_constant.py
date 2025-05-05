#date: 2025-05-05T17:06:43Z
#url: https://api.github.com/gists/da9946f43e53ecf14dc01058dbd7f668
#owner: https://api.github.com/users/definiteconfusion

import numpy as np

# First Matrix always horizontal
a = [
    0.1, 0.2, 0.5
]
b = [
    [0.2],
    [0.5],
    [0.7]
]
c = [
    [0.5],
    [0.7],
    [0.9]
]

ax = np.dot(a, b)
bx = np.dot(c, ax)

me = np.mean((c - bx))
print(f"ME: {round(np.sqrt(me)*100, 2)}%")