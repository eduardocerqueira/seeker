#date: 2024-06-21T17:04:07Z
#url: https://api.github.com/gists/9f663333ae648826aa091e12b40c0737
#owner: https://api.github.com/users/shadden

import numpy as np
from celmech.miscellaneous import sk,Dsk
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict
import sympy as sp

# choose j:j-k MMR
j = 5
k = 1
alpha = ((j-k)/j)**(2/3)
ecross = 1/alpha-1

# Exact disturbing function coefficient
C = df_coefficient_C(j,k-j,-k,0,0,0,*(0 for _ in range(4)))
nC = evaluate_df_coefficient_dict(C,alpha)

# Eqn 28 of HL18
x=sp.symbols('x',real=True)
exprn = 2*sp.besselk(0,2*k / sp.S(3)*( 1+ x/2)) * sp.exp(-2 * k * x / 3)/sp.pi / sp.factorial(k)
sk_by_yk_approx = sp.diff(exprn,x,k).xreplace({x:0}).evalf()

y0=0.01
print(nC,sk(k,y0)/(y0*ecross)**k, sk_by_yk_approx/ecross**k)