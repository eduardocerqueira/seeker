#date: 2021-10-26T16:55:20Z
#url: https://api.github.com/gists/f2a353e8d3fc788bac69c77f4aa298ec
#owner: https://api.github.com/users/b0noI

import numpy as np

def f(x):
  return x * 2 + 1

rng = np.random.default_rng(2021)
X = rng.random(1000)
Y = [f(x) for x in X]
