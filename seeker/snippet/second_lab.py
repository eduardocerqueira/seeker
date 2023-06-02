#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import math


def lagrange_interpolation(x, y, xi):
    n = len(x)
    yi = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (xi - x[j]) / (x[i] - x[j])
        yi += term
    return yi


def newton_first_order_interpolation(x, y, xi):
    n = len(x)
    diff_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (
                x[i + j] - x[i]
            )

    yi = diff_table[0][0]
    term = 1.0
    for j in range(1, n):
        term *= xi - x[j - 1]
        yi += term * diff_table[0][j]

    return yi


def newton_second_order_interpolation(x, y, xi):
    n = len(x)
    diff_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (
                x[i + j] - x[i]
            )

    yi = diff_table[0][0]
    term = 1.0
    for j in range(1, n):
        term *= xi - x[j - 1]
        yi += term * diff_table[0][j]
        term *= (xi - x[j]) / (x[j - 1] - x[j])
        yi += term * diff_table[0][j]

    return yi


def f(x):
    return x - math.exp(2 * x ** 2 - x ** 4 - 2)


def main():
    # Generate the data points
    x = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    y = [f(val) for val in x]

    # Interpolate at a specific point
    xi = 0.5

    # Perform interpolation
    lagrange_result = lagrange_interpolation(x, y, xi)
    newton_first_order_result = newton_first_order_interpolation(x, y, xi)
    newton_second_order_result = newton_second_order_interpolation(x, y, xi)

    # Print the results
    print(
        f"Lagrange interpolation result: {lagrange_result}, f({xi}) = {f(xi)}, error = {lagrange_result - f(xi)}"
    )
    print(
        f"Newton's first-order interpolation result: {newton_first_order_result}, f({xi}) = {f(xi)}, error = {newton_first_order_result - f(xi)}"
    )
    print(
        f"Newton's second-order interpolation result: {newton_second_order_result}, f({xi}) = {f(xi)}, error = {newton_second_order_result - f(xi)}"
    )


if __name__ == "__main__":
    main()
