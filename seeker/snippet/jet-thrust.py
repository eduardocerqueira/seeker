#date: 2023-04-05T17:09:06Z
#url: https://api.github.com/gists/2798ce4ad47d7648f73f54a3c83f9066
#owner: https://api.github.com/users/cshimmin

import numpy as np
import scipy.optimize

def get_thrust(pt):
    # pt should have shape (*, N, 2), for the px and py of an arbitrary batch-shape (*,)
    # of input jets with up to N particles. Zero-pad along the N dimension for jets that
    # have fewer than N variables.
    
    # start with a guess of alpha by taking the axis of the pT sum over the jet
    ptsum = np.sum(pt, axis=-2) # shape: (*, 2)
    alphas_0 = np.arctan2(ptsum[...,1], ptsum[...,0]) # shape: (*,)
    
    # define optimization objective:
    def obj(alphas):
        alphas = alphas[...,None] # shape: (*,) -> (*, 1)
        
        dotproduct = (pt[...,0]*np.cos(alphas))**2 + (pt[...,1]*np.sin(alphas))**2 # shape: (*,N)
        # note, no need to sqrt for optimizing
        numerator = dotproduct.sum(axis=-1) # shape: (*,)
        # and we have to take the mean over all jets to
        # to a scalar output, but the jacobian should be diagonal.
        out = numerator.mean()
        return -out # negative for minimize()
    
    # maximize the numerator:
    result = scipy.optimize.minimize(obj, alphas_0)
    
    #print(result)
    
    # now calculate thrust for the solved values of alphas:
    alphas = result.x[...,None] # shape: (*,) -> (*,1)
    
    dotproduct = (pt[...,0]*np.cos(alphas))**2 + (pt[...,1]*np.sin(alphas))**2 # shape: (*,N)
    numerator = np.sqrt(dotproduct).sum(axis=-1) # shape: (*,)
    
    ptmag = np.linalg.norm(pt, axis=-1) # shape: (*,N)
    denominator = ptmag.sum(axis=-1) # shape: (*,)
    
    return np.where(denominator > 0, numerator / denominator, 0) # shape: (*,)