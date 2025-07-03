#date: 2025-07-03T17:13:22Z
#url: https://api.github.com/gists/e8d3d6b89c2d4342fd72135019f3db6f
#owner: https://api.github.com/users/sahasatvik

#!/usr/bin/env python3

import sys
from math import sqrt

N = int(sys.argv[1])

while (N % 2) == 0:
    N /= 2
    print(2)

while (N % 3) == 0:
    N /= 3
    print(3)

p = 5
while N > 1 and p <= sqrt(N) + 1:
    while (N % p) == 0:
        N //= p
        print(p)
    p += 2
    while (N % p) == 0:
        N //= p
        print(p)
    p += 4

if N > 1:
    print(N)