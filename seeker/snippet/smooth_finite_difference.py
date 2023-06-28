#date: 2023-06-28T17:08:11Z
#url: https://api.github.com/gists/a834acd1cfe3b6563d9bdffd5e314bcd
#owner: https://api.github.com/users/twhentschel

import numpy as np
from scipy import ndimage

def smooth_derivative(f, x, order=1):
    """
    Numerical derivative, with a twist.
    We actually convolve f with the nth-order derivative of a gaussian kernel,
    which is equivalent to the nth order derivative of (f convolved with a plain gaussian),
    so there is some smoothing.
    The alternative is your typical finite differences method, which can be noisy for noisy input.
    
    From: @askewchan, https://stackoverflow.com/questions/18991408/python-finite-difference-functions
    _______
    f : (n, ) ndarray
      The function f evaluated at points x.
    x : (n, ) ndarray
      The indepedent points where f is evaluated.
    order : int, default is 1
      The number of derivative of f. If order=3, returns the 3rd-order derivative.
      
    returns: (n, ) ndarray
       The order'th-order finite difference of f
    """
    # x array is extended to keep same number of points.
    # uses wrapping format to agree with the gaussian filter
    dx = np.diff(np.append(x, [x[0]]))
    dnx = dx**order
    gf = ndimage.gaussian_filter1d(f, sigma=1, order=order, mode='wrap')/dnx
    
    return gf