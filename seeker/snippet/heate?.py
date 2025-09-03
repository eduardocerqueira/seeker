#date: 2025-09-03T16:51:19Z
#url: https://api.github.com/gists/a07c8f5040479819b603ffaf030810a9
#owner: https://api.github.com/users/v1nam

from matplotlib import pyplot as plt
import random
import math


n = 10
iters = list(range(200))
ls = sorted([0] + [random.uniform(0, 1) for _ in range(n-1)] + [1])
cls = [i*(1/n) for i in range(n+1)]

cf = lambda : sum(abs(cls[i] - ls[i]) for i in range(len(ls)))

def f():
    for i in range(1, len(ls) - 1):
       ls[i] = (ls[i-1] + ls[i+1]) / 2

def run():
    costs = []
    for j in iters:
        f()
        cost = cf()
        costs.append(cost)
    return costs

vals = run()
plt.plot(iters, list(map(math.log, vals)), label="log(cost)")
plt.show()