#date: 2023-04-11T17:00:46Z
#url: https://api.github.com/gists/74607799273631c9295d97cbf237c748
#owner: https://api.github.com/users/Umut-Can-Physics

import numpy as np
# Define Harper-Hofstadter Matrix
def H(p,q,kx,ky):
    # Define magnetic flux per unit-cell
    alpha = p/q
    # qxq size of zero matrix
    M = np.zeros((q,q), dtype=complex)
    for i in range(0,q):
        # Ortogonal elements of matris
        M[i,i] = 2*np.cos(ky-2*np.pi*alpha*i)
        # Other elements
        if i==q-1: 
            M[i,i-1]=1
        elif i==0: 
            M[i,i+1]=1
        else: 
            M[i,i-1]=1
            M[i,i+1]=1
    # Bloch condition
    if q==2:
        M[0,q-1] = 1+np.exp(-q*1.j*kx)
        M[q-1,0] = 1+np.exp(q*1.j*kx)
    else:
        M[0,q-1] = np.exp(-q*1.j*kx)
        M[q-1,0] = np.exp(q*1.j*kx)
    return M