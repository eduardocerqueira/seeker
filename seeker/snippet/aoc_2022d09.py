#date: 2022-12-09T17:02:15Z
#url: https://api.github.com/gists/11010c9f47630e20ed957de760b65f55
#owner: https://api.github.com/users/atant1

import os
from collections import defaultdict, Counter

with open(os.path.dirname(__file__) + "/in", "r") as f:
    data = [[(s := l.strip().split())[0], int(s[1])] for l in f.readlines()]

H, T, position = [0, 0], [0, 0], defaultdict(int)
move = {"R": (1, 0), "U": (0, 1), "L": (-1, 0), "D": (0, -1)}

for d, n in data:
    for i in range(n):
        H = [a + b for (a, b) in zip(H, move[d])]  # move H
        for k in range(1, 10):
            if not (abs(T[0] - H[0]) <= 1 and abs(T[1] - H[1]) <= 1):
                T = [
                    a - b for (a, b) in zip(H, move[d])
                ]  # move T, follow H when direction correct

            position[tuple(T)] = 1

print(len(position))

H, T, position = [0, 0], [[0, 0] for _ in range(10)], defaultdict(int)


def f_follow(h, t):
    return [int(t[i]+v/2) if abs(v:=h[i]-t[i])==2 else t[i]+v for i in [0,1] ]

for d, n in data:
    for i in range(n):
        H = [a + b for (a, b) in zip(H, move[d])]  # move H
        T[0] = H
        for k in range(1, 10):
            if not (abs(T[k - 1][0] - T[k][0]) <= 1 and abs(T[k - 1][1] - T[k][1]) <= 1):
                T[k] = f_follow(T[k - 1], T[k])

            position[tuple(T[k])] = (k if k >= position[tuple(T[k])] else position[tuple(T[k])])  # add T to dict

print(Counter(position.values())[9])
