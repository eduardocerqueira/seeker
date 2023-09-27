#date: 2023-09-27T17:00:27Z
#url: https://api.github.com/gists/69236e1f0e8ff5ab9089e7a9e32238d0
#owner: https://api.github.com/users/jlmcmchl

from sympy import *
import pylab as plt
import numpy as np

init_printing()

V, kV, kA, t, x0, xf, v0, vf, c1, c2, v = symbols('V, kV, kA, t, x0, xf, v0, vf, C1, C2, v')

true_values = [
    (V, 12),
    (kV, 2.5629),
    (kA, 0.43277),
    (x0, 0),
    (v0, -4),
    (xf, 10),
    (vf, -1)
]

x = symbols('x', cls=Function)

diffeq = Eq(kA * x(t).diff(t, t) + kV * x(t).diff(t), V)

soln = dsolve(diffeq)

x = soln.rhs
dx = x.diff(t)
ddx = dx.diff(t)


# get rid of c1, c2
x1 = x.subs([
    (c1, solve(Eq(x.subs(t, 0), x0), c1)[0]),
    (c2, solve(Eq(dx.subs(t, 0), v0), c2)[0])
])
x2 = x.subs([
    (c1, solve(Eq(x.subs(t, 0), xf), c1)[0]),
    (c2, solve(Eq(dx.subs(t, 0), vf), c2)[0]), 
    (V, -V)
])

dx1 = x1.diff(t)
dx2 = x2.diff(t)

t1_eqn = solve(Eq(dx1, v), t)[0]
t2_eqn = solve(Eq(dx2, v), t)[0]

equality = Eq(x2.subs(t, t2_eqn) - x1.subs(t, t1_eqn), 0).expand()

# move constant terms to rhs
equality = Eq(equality.lhs - xf + x0, equality.rhs - xf + x0)
equality = Eq(equality.lhs + kA*v0/kV, equality.rhs + kA*v0/kV)
equality = Eq(equality.lhs - kA*vf/kV, equality.rhs - kA*vf/kV)

equality = simplify(equality)

# move multiplicative terms, flip sign
equality = Eq(-equality.lhs * kV * kV / V / kA, -equality.rhs * kV * kV / V / kA)

# get rid of logarithm
equality = Eq(exp(equality.lhs), exp(equality.rhs))

# invert and move numerator terms
equality = Eq((V - kV*v0)*(V + kV*vf) / equality.lhs, (V - kV*v0)*(V + kV*vf) / equality.rhs)


# solve, take positive result
v_soln = solve(equality, v)[1]


inflection_t = solve(Eq(dx2, v_soln), t)[0]
inflection_distance = x2.subs(t, t2_eqn).subs(v, v_soln)

equality = simplify(Eq(dx1, v_soln))

# print(equality)
# move additive terms
equality = Eq(expand(equality.lhs) - V/kV, equality.rhs - V/kV)
equality = simplify(equality)

# move multiplicative terms
equality = Eq(equality.lhs * kV / (-V + kV*v0), equality.rhs * kV / (-V + kV*v0))

# take log
# sympy doesn't like this but we know lhs = -kV/kA*t
equality = Eq(-kV/kA*t, log(equality.rhs))

# move multiplicative terms, flip sign
equality = Eq(-equality.lhs * kA/kV, -equality.rhs * kA/kV)

first_t = equality.rhs
second_t = -inflection_t # negative because x2(0) = xf
print(f"total time = {first_t.subs(true_values) + second_t.subs(true_values)}")
print(f"inflection point: t={first_t.subs(true_values)} x={inflection_distance.subs(true_values)}, v={v_soln.subs(true_values)}")
