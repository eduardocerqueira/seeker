#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import math


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

def derivative(f, x, y, xi, h=1e-6):
    return (f(x, y, xi+h)-f(x, y, xi))/h

def f(x):
    return x - math.exp(2 * x ** 2 - x ** 4 - 2)

def df(x):
    return  4 * math.exp(-x**4 + 2*x**2 - 2)*x*(x**2 - 1) + 1


def main():
    # Generate the data points
    x = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    y = [f(val) for val in x]

    # Interpolate at a specific point
    xi = 0

    derivative_newton = derivative(newton_first_order_interpolation, x, y, xi)

    print(
        f"Derivative: {derivative_newton}, analytical value: {df(xi)}, error {df(xi) - derivative_newton}"
    )

if __name__ == "__main__":
    main()
