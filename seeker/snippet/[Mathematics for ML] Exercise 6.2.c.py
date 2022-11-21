#date: 2022-11-21T13:48:44Z
#url: https://api.github.com/gists/c24c303e456b80621ded2a9c4181cece
#owner: https://api.github.com/users/qjatn0120

import numpy as np
from random import random

class DensityFunction:

	def __init__(self, mu, sigma):
		self.mu = mu
		self.div = 2 * np.pi * np.sqrt(np.linalg.det(sigma))
		self.inv = np.linalg.inv(sigma)

	def prob(self, x):
		res = np.matmul(np.transpose(x - self.mu), self.inv)
		res = np.matmul(res, x - self.mu)
		res = np.exp(res / -2)
		return (res / self.div).item()

mu1 = np.array([[10], [2]]).astype(np.float64)
sigma1 = np.array([[1, 0], [0, 1]]).astype(np.float64)

mu2 = np.array([[0], [0]]).astype(np.float64)
sigma2 = np.array([[8.4, 2.0], [2.0, 1.7]]).astype(np.float64)

f1 = DensityFunction(mu1, sigma1)
f2 = DensityFunction(mu2, sigma2)

val = np.array([[10], [2]]).astype(np.float64)
step = 1
rep = 0

for i in range(1000000):
	theta = random() * 2 * np.pi
	nval = val.copy()
	nval[0][0] += np.cos(theta) * step
	nval[1][0] += np.sin(theta) * step
	rep += 1
	if(0.4 * f1.prob(val) + 0.6 * f2.prob(val) < 0.4 * f1.prob(nval) + 0.6 * f2.prob(nval)):
		val = nval
		rep = 0

	if rep == 100:
		rep = 0
		step *= 0.99

print(step, val)