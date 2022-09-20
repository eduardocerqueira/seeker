#date: 2022-09-20T17:03:36Z
#url: https://api.github.com/gists/1c69af956b46ef88c3c22a0dfaded0a3
#owner: https://api.github.com/users/paramjeetsharma

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
