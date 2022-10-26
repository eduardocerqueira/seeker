#date: 2022-10-26T17:09:24Z
#url: https://api.github.com/gists/9e1c858e701478ea22ad4412e4aee3e5
#owner: https://api.github.com/users/PershingSquare

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import svds

N = 2**25

diag_scale = np.ones((N,), np.float32)
diag_scale[-1] = 100

A = LinearOperator((N, N),
        matvec=lambda x: x.ravel() * diag_scale,
        rmatvec=lambda x : x.ravel() * diag_scale,
        dtype=np.float32)

print(svds(A, k=1, solver='lobpcg', return_singular_vectors=False)[0])