#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import math


def hermite_interpolation(x, y, xi):
    n = len(x)
    h = [0.0] * (2 * n)
    d = [0.0] * (2 * n)
    a = [0.0] * (2 * n)

    for i in range(n):
        j = 2 * i
        h[j] = x[i]
        h[j + 1] = x[i]
        d[j] = y[i]
        d[j + 1] = y[i]
        a[j] = calculate_slope(x, y, i)
        if i != 0:
            a[j - 1] = (d[j] - d[j - 2]) / (h[j] - h[j - 2])

    for i in range(2, 2 * n):
        for j in range(2 * n - 1, i - 1, -1):
            a[j] = (a[j] - a[j - 1]) / (h[j] - h[j - i])

    yi = d[0]
    term = 1.0
    for i in range(1, 2 * n):
        term *= xi - h[i - 1]
        yi += term * a[i]

    return yi


def calculate_slope(x, y, i):
    n = len(x)
    if i == 0:
        return (y[1] - y[0]) / (x[1] - x[0])
    elif i == n - 1:
        return (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
    else:
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        d1 = (y[i] - y[i - 1]) / h1
        d2 = (y[i + 1] - y[i]) / h2
        d1_term = (h1 + h2) / (h1 * (h1 + h2))
        d2_term = (h1 + h2) / (h2 * (h1 + h2))
        return d1 * d1_term + d2 * d2_term


def f(x):
    return x - math.exp(2 * x ** 2 - x ** 4 - 2)


def main():
    x = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    y = [f(val) for val in x]

    xi = -2

    hermite_result = hermite_interpolation(x, y, xi)

    # Print the result
    print(
        f"Hermite interpolation result: {hermite_result}, f({xi}) = {f(xi)}, error = {hermite_result - f(xi)}"
    )


if __name__ == "__main__":
    main()
