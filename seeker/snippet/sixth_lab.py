#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import math


def f(x):
    return math.cos(x ** 3)


def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral_sum = (f(a) + f(b)) / 2.0

    for i in range(1, n):
        x = a + i * h
        integral_sum += f(x)

    integral_approx = h * integral_sum
    return integral_approx


def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even.")

    h = (b - a) / n
    integral_sum = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        coefficient = 4 if i % 2 != 0 else 2
        integral_sum += coefficient * f(x)

    integral_approx = h / 3 * integral_sum
    return integral_approx


def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    integral_sum = 0.0

    for i in range(n):
        x_mid = a + (i + 0.5) * h
        integral_sum += f(x_mid)

    integral_approx = h * integral_sum
    return integral_approx


def main():
    a = 0.0
    b = math.pi / 2
    n = 1000

    # Wolfram result
    wolfram_approx = 0.701832097386658280444134

    trapezoidal_approx = trapezoidal_rule(f, a, b, n)
    simpsons_approx = simpsons_rule(f, a, b, n)
    midpoint_approx = midpoint_rule(f, a, b, n)

    print(
        f"Approximate integral using Trapezoidal Rule: {trapezoidal_approx}, error form wolfram: {trapezoidal_approx - wolfram_approx}"
    )
    print(
        f"Approximate integral using Simpson's Rule: {simpsons_approx}, error form wolfram: {simpsons_approx - wolfram_approx}"
    )
    print(
        f"Approximate integral using Midpoint Rule: {midpoint_approx}, error form wolfram: {midpoint_approx - wolfram_approx}"
    )


if __name__ == "__main__":
    main()
