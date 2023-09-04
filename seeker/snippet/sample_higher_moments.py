#date: 2023-09-04T17:05:59Z
#url: https://api.github.com/gists/80aef2f62f4ba2833bd9ffb7fd51df8b
#owner: https://api.github.com/users/rrlindsey

import statsmodels.sandbox.distributions.extras as extras
import scipy.interpolate as interpolate
import numpy as np

def generate_normal_four_moments(mu, sigma, skew, kurt, size=1000, sd_wide = 10):
    f = extras.pdf_mvsk([mu, sigma, skew, kurt])
    x = np.linspace(mu - sd_wide * sigma, mu + sd_wide * sigma, num=500)
    y = [f(i) for i in x]
    yy = np.cumsum(y) / np.sum(y)
    inv_cdf = interpolate.interp1d(yy, x, fill_value="extrapolate")
    rr = np.random.rand(size)
    return inv_cdf(rr)