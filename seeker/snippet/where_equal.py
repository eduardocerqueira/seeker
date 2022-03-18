#date: 2022-03-18T16:59:20Z
#url: https://api.github.com/gists/94286f7fa497595062271bb8925262ab
#owner: https://api.github.com/users/chmendoza

def where_equal(x, y):
    """ Find indices in `x` where `x`==`y`, with `y` being a shorter array

    Parameters
    ----------
    x: numpy.array
        Reference 1D array. x.shape=(N,). `x` is assumed to be sorted in ascending order.
    y: numpy.array
        Query 1D array. y.shape=(m,). N>m. `y` is assumed to have unique (non-repeating) values.
    
    Returns
    -------
    idx: numpy.array
        Indices of `x` where values of `x` are in `y`
    
    Notes
    -----
    A naive for-loop implementation will be O(N*m). Since searchsorted uses binary search, 
    I believe the cost here is O(log(N)*m)?
    
    References:
    https://stackoverflow.com/q/18875970/4292705
    https://stackoverflow.com/a/4708737/4292705
    """

    start = np.searchsorted(x, y, side='left')
    end = np.searchsorted(x, y, side='right')
    idx = np.r_[tuple(slice(s, e) for s, e in zip(start, end))]

    return idx