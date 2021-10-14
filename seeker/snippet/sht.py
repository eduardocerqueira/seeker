#date: 2021-10-14T17:10:38Z
#url: https://api.github.com/gists/59c574a7502f843b9d1a7c746c777b3b
#owner: https://api.github.com/users/maedoc

import numpy as np
from scipy.special import sph_harm

class ShtDiff:

    def __init__(self, lmax=16, nlat=64, D=-1e-2, nlon=None):
        self.lmax = lmax
        self.nlat = nlat
        self.D = D
        self.nlon = nlon or (2 * nlat)
        self.setup_arrays()
    
    def setup_arrays(self):
        points, weights = np.polynomial.legendre.leggauss(self.nlat)
        self.gauss_weights = weights[::-1]
        self.phi_grid = np.arcsin(points) + np.pi/2
        # Legendre
        L = []
        for m in range(self.lmax):
            l = np.r_[m:(self.lmax+1)]
            fwd = self.gauss_weights[None, :] * sph_harm(m, l[:, None], 0, self.phi[None, :]).conjugate()
            bwd = sph_harm(m, l[None, :], 0, self.phi[:, None])
            dll = self.D * l * (l + 1)
            L.append(bwd.dot(dll[:, None] * fwd))
        self.legendre = np.array(L)
        # Fourier
        C = np.cos(2*np.pi/self.nlon*np.c_[:self.lmax]*np.r_[:self.nlon])
        S = -np.sin(2*np.pi/self.nlon*np.c_[:self.lmax]*np.r_[:self.nlon])
        self.fourier = (C + 1j*S)

    def __call__(self, x):
        self.fourier.T.dot(
            np.einsum(
                'acb,ab->ac', 
                self.legendre, 
                self.fourier.dot(x.T))).real.T / self.nlon