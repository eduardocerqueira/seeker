#date: 2021-12-17T16:56:15Z
#url: https://api.github.com/gists/995d52cd30b93a5537cab7a31523e276
#owner: https://api.github.com/users/ad-1

import math
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

t, a = sp.symbols('t a')


def evaluate_terms(terms, point):
    """
    terms: list of polynomial terms
    point: point (a) of approximation
    """
    return [terms[n].subs({a: point}) / math.factorial(n) * (t - point) ** n for n in range(len(terms))]


def taylor_polynomial(func, order, point, derivatives=None):
    """
    find nth order Taylor polynomial
    func: symbolic function to approximate
    order: highest order derivative for Taylor polynomial (interger)
    point: point about which the function is approximated
    derivatives: list of Taylor terms
    """

    # initialize list of derivatives
    if derivatives is None:
        derivatives = [func.subs({t: a})]

    # check if highest order derivative is reached
    if len(derivatives) > order + 1:
        # return list of derivatives
        return derivatives, evaluate_terms(derivatives, point)

    # differentiate function with respect to t
    derivative = func.diff(t)

    # append to list of symbolic derivatives ** substitute t with a **
    derivatives.append(derivative.subs({t: a}))

    # recursive call to find next term in Taylor polynomial
    return taylor_polynomial(derivative, order, point, derivatives)


if __name__ == '__main__':

    # analysis label
    label = 'cos(t)'

    # symbolic function to approximate
    f = sp.cos(t)

    # point about which to approximate
    approximation_point = 0

    # definte time start and stop
    start = -sp.pi
    stop = sp.pi
    time = np.arange(start, stop, 0.01)

    # find taylor polynomial terms describing function f(t)
    symbolic_derivatives, taylor_terms = taylor_polynomial(func=f, order=4, point=approximation_point)

    print('derivatives:', symbolic_derivatives)
    print('taylor terms:', taylor_terms)

    # initialize plot
    fig, ax = plt.subplots()
    ax.set(xlabel='t', ylabel='f(t)', title=f'Taylor Polynomial Approximation: {label}')
    legend = []

    # initialize taylor polynomial
    polynomials = []
    poly = None

    # loop through tayloer terms
    for term in range(len(taylor_terms)):
        # build up polynomial on each iteration
        poly = taylor_terms[term] if poly is None else poly + taylor_terms[term]
        polynomials.append(poly)

        # plot current polynomial approximation
        ax.plot(time, [poly.subs({t: point}) for point in time])

        # append item to legend
        legend.append(f'P{term}')

    # plot actual function for comparison
    ax.plot(time, [f.subs({t: point}) for point in time])
    legend.append(f'f(t)')

    # create dataframe
    df = pd.DataFrame(
        {'symbolic_derivatives': symbolic_derivatives,
         'taylor_terms': taylor_terms,
         'polynomials': polynomials
         })

    # save and show results
    ax.legend(legend)
    ax.grid()
    plt.show()
    plt.savefig(f'taylor_{label}.png')
    df.to_csv(f'taylor_{label}.csv', encoding='utf-8')
    print(df.head())
