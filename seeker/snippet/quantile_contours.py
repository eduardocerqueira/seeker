#date: 2024-05-31T16:47:55Z
#url: https://api.github.com/gists/ae5deca9aca48b2904f62c6d9372aeef
#owner: https://api.github.com/users/petrelharp

# Draw 'quantile contours': for instance, the 0.9 contour
# encloses roughly 90% of the points.

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

x = 2 * rng.normal(size=10000)
y = rng.normal(size=10000)

kde = scipy.stats.gaussian_kde(np.vstack([x, y]))
X, Y = np.meshgrid(np.linspace(-6, 6, 101), np.linspace(-3,3,61))
XY = np.vstack([X.ravel(), Y.ravel()])
Z = kde(XY).reshape(X.shape)


z = Z.ravel()
z += np.min(z)
z /= np.sum(z)
iz = np.argsort(z)
z[iz] = np.cumsum(z[iz])
cZ = z.reshape(Z.shape)

levels = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
fig, ax = plt.subplots()
# ax.contourf(X, Y, cZ, levels=levels)
ax.hexbin(x, y)
cs = ax.contour(X, Y, cZ, levels=levels)
ax.clabel(cs, levels, inline=True, fmt=lambda x: f"{x:.2}", fontsize=12)
ax.set_aspect(1)