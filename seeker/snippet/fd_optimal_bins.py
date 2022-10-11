#date: 2022-10-11T17:21:54Z
#url: https://api.github.com/gists/058d1b318937369ac539efd453964489
#owner: https://api.github.com/users/maxmarkov

import numpy as np
import math 

def fd_optimal_bins(data: np.array) -> int:
    """ The Freedman-Diaconis rule for optimal bin selection
    Parameters: 
        data (np.array) - a one-dimensional array with data
    Returns:
        nbins (int) - number of bins
    """
    assert data.ndim == 1
    n = data.size
    
    p25, p75 = np.percentile(data, [25, 75])

    width = 2. * (p75 - p25)/n**(1./3)
    nbins = math.ceil((data.max() - data.min()) / width)
    nbins = max(1, nbins)
    
    return nbins

nbins = fd_optimal_bins(data)
plot_histogram(data, bins=nbins)