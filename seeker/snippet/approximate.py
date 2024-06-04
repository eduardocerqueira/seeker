#date: 2024-06-04T17:07:17Z
#url: https://api.github.com/gists/33dde9916f0c04404b42888393458af3
#owner: https://api.github.com/users/eebmagic

import random
import math

def genPoints(N):
    points = []
    for i in range(N):
        points.append((random.random(), random.random()))

    return points

def countInCircle(points):
    inside = 0
    for point in points:
        x, y = point
        dist = math.sqrt(x**2 + y**2)

        if dist < 1:
            inside += 1

    return inside


N = 1_000_000
points = genPoints(N)
print(f'with {N:,} points')

ins = countInCircle(points)
approx = 4 * ins / N
print(f'\n             inside: {ins:,}')
print(f'\napproximation of pi: {approx}')

error = math.pi - approx
print(f'\n              error: {error}')
