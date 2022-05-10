#date: 2022-05-10T17:10:51Z
#url: https://api.github.com/gists/2a6b089c8223bca1772abbde6ff63124
#owner: https://api.github.com/users/rpromoditha

import matplotlib.pyplot as plt
plt.figure(figsize=[7, 5])

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=None, 
                  random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='plasma')
plt.savefig("make_moons.png")