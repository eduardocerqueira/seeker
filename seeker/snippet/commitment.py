#date: 2023-01-06T16:48:21Z
#url: https://api.github.com/gists/fb666ec7ccb334e1660b1041224735ad
#owner: https://api.github.com/users/0xNineteen

from hashlib import sha256
import bitcoin as btc
import random
import numpy as np 

def rand():
    return random.randint(1, 20)

def ecc_vector_sum(V, v):
    assert len(V) == len(v)
    C = btc.fast_multiply(V[0], v[0])
    for i in range(1, len(v)):
        C = btc.fast_add(
            C, 
            btc.fast_multiply(V[i], v[i])
        )
    return C

def vector_set_commitment():
    # want to commit to this set of vector
    x = [[1, 2, 3],
        [2, 3, 4]]

    H = btc.fast_multiply(btc.G, rand())
    r = [rand() for _ in range(len(x))] 
    G = [btc.G for _ in range(len(x[0]))]

    # start of interactive proof (random vector commitment C_0)
    r_0, x_0 = rand(), list(np.random.randint(0, 20, size=(len(x[0]),)))
    r.insert(0, r_0)
    x.insert(0, x_0)

    # commitment
    C = [ecc_vector_sum(
            [H] + G, 
            [r[i]] + x[i]
        )
        for i in range(len(x))]

    # challenge
    e = rand()

    # proof 
    # G x = [[G0 1, G1 2, G2 3],
    #     [G0 2, G1 3, G2 4]]
    # G x e**i = [[G0 1 * 1, G1 2, G2 3],
    #     [G0 2 * e, G1 3 * e, G2 4 * e]]
    # G x e**i = [[G0 (1 * 1 + 2e), G1 (2 + 3e), G2 (3 + 4e)]]
    # G x e**i = G [[1 + 2e, 2 + 3e, 3 + 4e]]

    # ie, in matrix multiplication form:
    # coefficients = [ [1], 
    #                  [e],
    #                  [e**2]]
    coefficients = np.array([e ** i for i in range(len(x[0]))])
    z = list(coefficients @ x)
    s = sum([r[i] * e ** i for i in range(len(r))])

    rhs = ecc_vector_sum(
        [H] + G,
        [s] + z
    )
    lhs = ecc_vector_sum(
        C, 
        [e ** i for i in range(len(C))]
    )
    print(rhs == lhs)

if __name__ == '__main__':
    print('---')
    vector_set_commitment()