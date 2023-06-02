#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

import math

def fixed_point_iteration(g, x0, tol=1e-6, max_iterations=100):
    x = x0

    for i in range(max_iterations):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    print("Fixed-point iteration method did not converge within the given iterations.")
    return None


def newton_method(f, x0, tol=1e-6, max_iterations=100):
    x = x0

    for i in range(max_iterations):
        f_prime = (f(x + tol) - f(x - tol)) / (2 * tol)
        delta_x = f(x) / f_prime
        x -= delta_x

        if abs(delta_x) < tol:
            return x

    print("Newton's method did not converge within the given iterations.")
    return None


def secant_method(f, x0, x1, tol=1e-6, max_iterations=100):
    for i in range(max_iterations):
        delta_x = (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        x0 = x1
        x1 -= delta_x

        if abs(delta_x) < tol:
            return x1

    print("Secant method did not converge within the given iterations.")
    return None


def bisection_method(f, a, b, tol=1e-6, max_iterations=100):
    if f(a) * f(b) >= 0:
        print("Bisection method cannot guarantee convergence.")
        return None

    for i in range(max_iterations):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    print("Bisection method did not converge within the given iterations.")
    return None


# f(x) = x - e^(2x^2 - x^4 - 1)
def f(x):
    return x - math.exp(2 * x ** 2 - x ** 4 - 1)


# g(x) = e^(2x^2 - x^4 - 1)
def g(x):
    return math.exp(2 * x ** 2 - x ** 4 - 1)


def main():
    root_bisection = bisection_method(f, -10, 10)

    if root_bisection is not None:
        print(f"Bisection Method - Root found: {root_bisection}")
    else:
        print("Bisection Method - Failed to find a root.")

    root_newton = newton_method(f, -10)

    if root_newton is not None:
        print(f"Newton's Method - Root found: {root_newton}")
    else:
        print("Newton's Method - Failed to find a root.")

    root_secant = secant_method(f, -10, 10)

    if root_secant is not None:
        print(f"Secant Method - Root found: {root_secant}")
    else:
        print("Secant Method - Failed to find a root.")

    root_fixed_point = fixed_point_iteration(g, -10)

    if root_fixed_point is not None:
        print(f"Fixed-Point Iteration Method - Root found: {root_fixed_point}")
    else:
        print("Fixed-Point Iteration Method - Failed to find a root.")


if __name__ == "__main__":
    main()
