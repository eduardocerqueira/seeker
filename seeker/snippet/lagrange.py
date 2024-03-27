#date: 2024-03-27T17:09:06Z
#url: https://api.github.com/gists/2b20300662996088d3013be23a3a39b2
#owner: https://api.github.com/users/pvar0000

import numpy as np
import matplotlib.pyplot as plt


def L(x, i, z):
    n = len(x)
    result = 1
    for j in range(n):
        if j != i:
            result *= (z - x[j][0]) / (x[i][0] - x[j][0])
    return result


def P(x, z):
    n = len(x)
    result = 0
    for i in range(n):
        result += x[i][1] * L(x, i, z)
    return result


def f(x):
    return 1 / (1 + 20 * x**2)


# Parámetros para la simulación, nodos equidistantes
x_1 = [-0.83, -0.56, -0.28, 0.00, 0.28, 0.56]  # 6 nodos
x_2 = [-0.90, -0.70, -0.50, -0.30, -0.10, 0.10, 0.30, 0.50, 0.70, 0.90]  # 10 nodos
x_3 = [
    -0.92,
    -0.77,
    -0.62,
    -0.46,
    -0.31,
    -0.15,
    0.00,
    0.15,
    0.31,
    0.46,
    0.62,
    0.77,
]  # 12 nodos

data_1 = [(x, f(x)) for x in x_1]
data_2 = [(x, f(x)) for x in x_2]
data_3 = [(x, f(x)) for x in x_3]

z = np.linspace(-1, 1, 1000)

p_5 = P(data_1, z)
p_9 = P(data_2, z)
p_11 = P(data_3, z)

plt.plot(z, f(z), label="f(x)", color="black")
plt.plot(z, p_5, label="p_5(x)", color="red")
plt.plot(z, p_9, label="p_9(x)", color="green")
plt.plot(z, p_11, label="p_11(x)", color="blue")

plt.scatter(*zip(*data_1), color="red", label="Puntos de interpolación de grado 5")
plt.scatter(*zip(*data_2), color="green", label="Puntos de interpolación de grado 9")
plt.scatter(*zip(*data_3), color="blue", label="Puntos de interpolación de grado 11")

plt.title("Interpolación de Lagrange")
plt.grid(True)
plt.legend()
plt.xlim(-1, 1)
plt.ylim(-0.2, 1.2)
plt.show()
