#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import numpy as np
import math


def aitken_scheme(x, y, x_val):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (
                (x_val - x[i]) * table[i + 1, j - 1]
                - (x_val - x[i + j]) * table[i, j - 1]
            ) / (x[i + j] - x[i])

    return table[0, -1]


def f(x):
    return x - math.exp(2 * x ** 2 - x ** 4 - 2)


def main():
    x = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    y = [f(val) for val in x]

    xi = 2
    result = aitken_scheme(x, y, xi)

    print(
        f"Aitken scheme result: {result}, f({xi}) = {f(xi)}, error = {result - f(xi)}"
    )


if __name__ == "__main__":
    main()
