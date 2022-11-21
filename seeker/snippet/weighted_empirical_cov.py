#date: 2022-11-21T17:11:45Z
#url: https://api.github.com/gists/955e65c3e4b937597688c142dc9659a5
#owner: https://api.github.com/users/vene

"""Fitting a gaussian to an empirical weighted measure"""

# author: vlad niculae <v.niculae@uva.nl>
# license: bsd

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax


def main():

    n = 7
    d = 2

    # generate n embeddings in d dimensions

    rng = np.random.default_rng(42)
    V = rng.standard_normal(size=(n, d))

    plt.scatter(V[:, 0], V[:, 1], marker='.')

    p = np.zeros(n)

    # scenario 1: almost centered on one point
    p[6] = 7

    # scenario 2: split between two points
    # p[4] = 7
    # p[6] = 7

    # scenario 3: split between three points
    p[1] = 7
    p[4] = 7
    p[6] = 7

    p = softmax(p)

    # empirical mean and cov
    mu = np.dot(p, V)
    cov = (V - mu).T @ (p[:, np.newaxis] * (V - mu))
    u, s, _ = np.linalg.svd(cov)
    prec = u.T @ ((1/s)[:, np.newaxis] * u)
    sqrt_prec = (1 / np.sqrt(s))[:, np.newaxis] * u


    def energy(x1, x2):
        # assume x1, x2 have shape (a, b)
        x = np.stack([x1, x2])  # (2, a, b)
        x = x.transpose(1, 2, 0)  # (a, b, 2)

        ctr = x - mu  # (a, b, 2)
        maha = ((ctr @ sqrt_prec.T) ** 2).sum(axis=-1)
        return maha

        # if we want the entire logpdf:
        # log normalizing constant
        # logdet = logsumexp(s)
        # log_pdf = -log(2pi) - .5*logdet - .5 * maha
        # return np.log(2 * np.pi) + (logdet + maha) / 2

    # chi square with 2df at p=.5, p=.1, and p=0.001
    levels = [1.39, 4.61, 13.82]

    x_min = y_min = -2
    x_max = y_max = 2
    n_points = 100
    grid_x = np.linspace(x_min, x_max, n_points)
    grid_y = np.linspace(y_min, y_max, n_points)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    eg = energy(mesh_x, mesh_y)

    plt.contour(mesh_x, mesh_y, eg, levels=levels)
    plt.show()


if __name__ == '__main__':
    main()

