#date: 2022-02-03T17:01:06Z
#url: https://api.github.com/gists/3cb5cad065eb712e342798d022ec44a2
#owner: https://api.github.com/users/bolverk

import sympy

def eqn1_cross_check():

  gamma_s, rho, mu, x, omega, m = sympy.symbols('gamma_s rho mu x omega m', positive=True)

  sakurai = gamma_s - rho**(-mu)
  atmosphere = rho - x**omega
  mass = m - rho*x

  _ = sympy.solve([sakurai, atmosphere, mass], [gamma_s, rho, x], dict=True)[0]
  _ = gamma_s.subs(_)
  _ = _.subs(mu, 0.2)
  _ = [_.subs(omega,wv) for wv in [1.5,3]]
  return _

def eqn_2():

  from astropy.constants import (
    M_sun,
    R_sun,
    sigma_T,
    m_p
    )
  import numpy

  kappa, m, R = sympy.symbols('kappa m R')

  _ = 1 - kappa*m/R**2/4/numpy.pi
  _ = sympy.solve(_, m)[0]
  _ = sympy.lambdify((kappa, R),_)
  _ = _(sigma_T/m_p,R_sun)/M_sun
  return _