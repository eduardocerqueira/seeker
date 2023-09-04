#date: 2023-09-04T16:57:27Z
#url: https://api.github.com/gists/0b9c761ba64811eaefab11a88caeaad0
#owner: https://api.github.com/users/ahwillia

import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform
from math import comb

@numba.jit(nopython=True)
def index(n, i, j):
    """
    Computes linear index of (i, j) from the (n x n) distance matrix.
    """
    if j > i:
        return (j - i) + (i * ((n - 1) + (n - i)) // 2) - 1
    else:
        return (i - j) + (j * ((n - 1) + (n - j)) // 2) - 1

@numba.jit(nopython=True)
def alg31(n, d, e, z):
    """
    Implements Algorithm 3.1 from Brickell et al. (2008), passing over
    all triangles once.

    Brickell, J., Dhillon, I. S., Sra, S., & Tropp, J. A. (2008).
    The metric nearness problem. SIAM Journal on Matrix Analysis and
    Applications, 30(1), 375-396.
    """
    u = 0
    tt = 0

    # Iterate of elements (i, j) of the distance matrix.
    for i in range(n):
        for j in range(i + 1, n):

            # Compute linear index.
            ij = index(n, i, j)

            # Iterate over (i, k, j) for k not in (i, j).
            for k in range(n):
                if (i != k) and (j != k):

                    # Get linear index.
                    ik = index(n, i, k)
                    kj = index(n, k, j)

                    # Compute update
                    v = d[ik] + d[kj] - d[ij]
                    ts = (e[ij] - e[ik] - e[kj] - v) / 3
                    t = max(ts, -z[u])

                    # Keep running total of updates.
                    tt += abs(t)

                    # Apply update
                    e[ij] -= t
                    e[ik] += t
                    e[kj] += t
                    z[u] += t
                    u += 1
    
    return tt

@numba.jit(nopython=True)
def max_violation(D):
    """
    Returns the worst triangle inequality violation over all directed triplets
    in an (n x n) distance matrix. Negative numbers indicate a triangle
    inequality violation.
    """
    n = D.shape[0]
    v = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                v = min(v, D[i, k] + D[k, j] - D[i, j])
    return v

def metric_repair(D, num_iters=10, verbose=True):
    n = D.shape[0]
    idx = np.triu_indices_from(D, 1)
    d = D[idx]
    e = np.zeros_like(d)
    z = np.zeros(3 * comb(n, 3))
    for it in range(num_iters):
        tt = alg31(n, d, e, z)
        if verbose:
            print("Param update:", tt)
    D_clean = np.zeros_like(D)
    D_clean[idx] = d + e
    D_clean += D_clean.T
    return D_clean

if __name__ == "__main__":
    print("Test on metric data (no repair needed...)")
    X = np.random.randn(6, 3)
    D = squareform(pdist(X, 'euclidean'))
    D_clean = metric_repair(D, num_iters=3, verbose=True)
    print("FINAL VIOLATION:", max_violation(D_clean))

    print("\n\n\nTest on non-metric data...)")
    D = squareform(pdist(X, 'sqeuclidean'))
    D_clean = metric_repair(D, num_iters=20, verbose=True)
    max_violation(D_clean)
    print("FINAL VIOLATION:", max_violation(D_clean))