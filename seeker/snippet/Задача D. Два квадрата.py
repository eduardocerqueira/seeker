#date: 2023-01-02T16:59:43Z
#url: https://api.github.com/gists/df00f16f447b76bf5f04d77e1a81ce37
#owner: https://api.github.com/users/AspirantDrago

from math import sqrt, ceil, floor

s = int(input())
max_d1 = int(sqrt(s))
answer = None
eps = float('inf')
for d1 in range(1, max_d1):
    d2_float = sqrt(s - d1 ** 2)
    for d2 in (floor(d2_float), ceil(d2_float)):
        err = abs(s - d1 ** 2 - d2 ** 2)
        if err < eps:
            answer = d1, d2
            eps = err
print(*answer)
