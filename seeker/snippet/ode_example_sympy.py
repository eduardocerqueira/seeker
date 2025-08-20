#date: 2025-08-20T16:59:25Z
#url: https://api.github.com/gists/d3a9eb9dde2a9f2c23496d81ea6e45bd
#owner: https://api.github.com/users/Maxxum69

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

sym.init_printing()

# Integral calculation constants
a = 0
b = 20
h = 0.4

# Variables
t = sym.symbols('t')
y = sym.Function('y')

# Initial conditions
ics = {y(0): 1}

# Equation hands
left_hand = sym.Derivative(y(t), t)
right_hand = -5 * y(t)

eq = sym.Eq(left_hand, right_hand)
print(sym.pretty(eq))

# Solve equation
sol = sym.dsolve(eq, y(t), ics=ics)
print(sym.pretty(sol))

# Transform into function
fun_y = sym.lambdify(t, sol.rhs, modules=['numpy'])
print(fun_y(0), fun_y(1))

# Séries de données
t = np.arange(a, b + h, h)
y = fun_y(t)
print(t, y)

# Graphique de y(t)
plt.plot(t, y, color='b')
plt.xlim(0, 20)
plt.show()


# ================================== Ex 2

x = sym.var('x')
f = sym.Function('f')

diffeq = sym.Eq(sym.Derivative(f(x), x), x + f(x) / 5)
ics = {f(0): -3}

sol2 = sym.dsolve(diffeq, f(x), ics=ics).rhs

print(sym.pretty(sym.simplify(sol2)))


# ================================== Ex 3

f_2 = sym.Function('f_2')
x_2 = sym.var('x_2')

A = sym.var("A")

ics = {f_2(0): A}

sol3 = sym.dsolve(f_2(x_2).diff(x_2) - f_2(x_2), f_2(x_2), ics=ics)

print(sym.pretty(sol3))