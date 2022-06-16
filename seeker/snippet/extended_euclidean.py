#date: 2022-06-16T17:06:49Z
#url: https://api.github.com/gists/33038e6cbd3223da01df7379310dc4ce
#owner: https://api.github.com/users/aaaaaaaalesha

# Copyright 2022 aaaaaaaalesha
import math

from prettytable import PrettyTable

table = PrettyTable(field_names=('f', 'g', 'd'))

f, g = map(int, input("Введите 2 целых числа через пробел: ").split())
a_, b_, d_ = 1, 0, f
a0, b0, d0 = 0, 1, g
table.add_row((1, 0, f))
table.add_row((0, 1, g))

Qi = math.ceil(f / g)
ai, bi, di = a_ - Qi * a0, b_ - Qi * b0, d_ - Qi * d0

a_, b_, d_ = a0, b0, d0
a0, b0, d0 = ai, bi, di

table.add_row((ai, bi, di))

while di != 0:
    Qi = math.ceil(d_ / d0)
    ai, bi, di = a_ - Qi * a0, b_ - Qi * b0, d_ - Qi * d0

    a_, b_, d_ = a0, b0, d0
    a0, b0, d0 = ai, bi, di

    table.add_row((ai, bi, di))

print(table)
print(f"{a_} * {f} + {b_} * {g} = {d_}")
print(f"{f}⁻¹ = {a_} (mod {g})")
print(f"{g}⁻¹ = {b_} (mod {f})")
print(f"gcd({f}, {g}) = {d_}")

"""
example:
Введите 2 целых числа через пробел: 523 186
+-----+------+-----+
|  f  |  g   |  d  |
+-----+------+-----+
|  1  |  0   | 523 |
|  0  |  1   | 186 |
|  1  |  -3  | -35 |
|  5  | -14  |  11 |
|  16 | -45  |  -2 |
|  85 | -239 |  1  |
| 186 | -523 |  0  |
+-----+------+-----+
85 * 523 + -239 * 186 = 1
523⁻¹ = 85 (mod 186)
186⁻¹ = -239 (mod 523)
gcd(523, 186) = 1

Process finished with exit code 0
"""