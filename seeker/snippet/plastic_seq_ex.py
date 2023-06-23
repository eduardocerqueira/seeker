#date: 2023-06-23T16:59:00Z
#url: https://api.github.com/gists/b2bf17f347597086127b427d2696917a
#owner: https://api.github.com/users/firestrand

import numpy as np
from matplotlib import pyplot as plt


# Code started from here and modified to use numpy more efficiently:
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/


def phi(d: int) -> float:
    x = 2.0000
    for _ in range(10):
        x = pow(1. + x, 1. / (d + 1))
    return x


def numpy_impl(dim: int, n_samples: int):
    g = phi(dim)
    j = np.arange(dim)
    alpha = np.mod(np.power(1. / g, j + 1), 1)

    # This number can be any real number.
    # Common default setting is typically seed=0
    # But seed = 0.5 is generally better.
    seed = 0.5
    i = np.arange(n_samples) + 1
    z = np.mod(seed + alpha * i[:, np.newaxis], 1)
    # print(z)
    
    # hard coded display of 2d sequence, remove for dimensions != 2
    plt.scatter(z[:, 0], z[:, 1], color='b')
    plt.show()


if __name__ == '__main__':
    numpy_impl(2, 100)