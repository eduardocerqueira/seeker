#date: 2022-12-28T16:28:06Z
#url: https://api.github.com/gists/278a9a05a88dd1e9f7ed9fc7b91ff1bd
#owner: https://api.github.com/users/tchaumeny

# See https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/, chapter 30
# /!\ Do not use it in production /!\

from math import e, pi

import numpy as np
from numpy.testing import assert_array_equal


def fft(L, sign=-1):
    n = len(L)
    if n == 1:
        return L
    w_n = e**(sign * 2j * pi / n)
    w = 1
    F_even = fft(L[::2], sign)
    F_odd = fft(L[1::2], sign)
    F = [0] * n
    for k in range(n // 2):
        F[k] = F_even[k] + w * F_odd[k]
        F[k + n // 2] = F_even[k] - w * F_odd[k]
        w = w * w_n
    return F

def ifft(L):
    return [c / len(L) for c in fft(L, sign=1)]

def multiply(P, Q):
    n = 1
    while n < len(P) + len(Q) - 1:
        n <<= 1
    P = P + [0] * (n - len(P))
    Q = Q + [0] * (n - len(Q))
    F = [a * b for a, b in zip(fft(P), fft(Q))]
    return ifft(F)

def assert_equal(P, Q):
    assert_array_equal(np.trim_zeros(np.round(np.real(P)).astype(np.int32), "b"), Q)

assert_equal(multiply([1, 2, 3], [9, 5, 1]), [9, 23, 38, 17, 3])
assert_equal(multiply([1, 1, 1], [1, 1]), [1, 2, 2, 1])
assert_equal(multiply([1, 2, 3], [4, 5, 6, 7, 8, 9, 10]), [4, 13, 28, 34, 40, 46, 52, 47, 30])
