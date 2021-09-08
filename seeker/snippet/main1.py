#date: 2021-09-08T17:01:22Z
#url: https://api.github.com/gists/171e355db6e8edf138760b5efb0e597f
#owner: https://api.github.com/users/jaishilp

"""
Created on Tue Sep 7 2021

"""

import numpy as np

def angle ( u, v ):
    """
    Computes the angle between the vectors u and v
    """
    
    uv = np.dot( u, v )
    nu = np.sqrt( np.dot( u, u ))
    nv = np.sqrt( np.dot( v, v ))
    
    return np.arccos( uv/(nu*nv) )

def ortho_proj( v, r ):
    """
    Computes the orthogonal projection of the vector v
    onto the reference vector r
    """
    
    v_parallel = 
    
    return v_parallel

def scalar_proj( v, r ):
    """
    Computes the scalar orthogonal projection of the vector v
    onto the reference vector r
    """
    
    v_scalar = 
    
    return v_scalar

def perp_factor( v, r ):
    """
    Computes v_perp in the orthogonal decomposition:
    
    v = v_parallel + v_perp
    
    of the vector v with respect to the reference vector r
    """
    
    v_perp = 
    
    return v_perp