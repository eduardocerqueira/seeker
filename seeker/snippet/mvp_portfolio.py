#date: 2022-11-16T16:53:59Z
#url: https://api.github.com/gists/5bf53944ffd13696fb4102a77602378c
#owner: https://api.github.com/users/yhilpisch

#
# Mean-Variance Portfolio Class
# Markowitz (1952)
#
# Python for Asset Management
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import math
import numpy as np
import pandas as pd

def portfolio_return(weights, rets):
    return np.dot(weights.T, rets.mean()) * 252

def portfolio_variance(weights, rets):
    return np.dot(weights.T, np.dot(rets.cov(), weights)) * 252

def portfolio_volatility(weights, rets):
    return math.sqrt(portfolio_variance(weights, rets))
