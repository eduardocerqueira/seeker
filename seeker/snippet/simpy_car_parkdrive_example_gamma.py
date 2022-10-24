#date: 2022-10-24T17:09:41Z
#url: https://api.github.com/gists/d869372a357d3df5d2d24c80f503f62c
#owner: https://api.github.com/users/galenseilis

import simpy

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

y = []
t = []

def car(env):
    while True:
        print(f'Start parking at {env.now}')
        t.append(env.now)
        y.append(0)
        parking_duration = gamma.rvs(1, 1)
        yield env.timeout(parking_duration)

        print(f'Start driving at {env.now}')
        t.append(env.now)
        y.append(1)
        trip_duration = gamma.rvs(0.5, 1)
        yield env.timeout(trip_duration)

env = simpy.Environment()
env.process(car(env))

env.run(until=20)

plt.step(t, y, label='Y')
plt.plot(t, np.cumsum(y) / np.arange(1, len(y)+1), label='$\\mathbb{E}[Y_{n \\leq t}]$')
plt.xlabel('Time (t)')
plt.legend()
plt.show()
