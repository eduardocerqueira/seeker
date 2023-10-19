#date: 2023-10-19T17:06:02Z
#url: https://api.github.com/gists/5948255c612f12ef94539b8b816546ac
#owner: https://api.github.com/users/ccyehintx

import matplotlib.pyplot as plt
import numpy as np
import math


def dotrho(mat):
    # Generate the rho change with time
    N = len(mat)
    dtimemat = np.zeros((N,N))
    dtimemat[0][0] = D*(mat[0][1] + mat[1][0] - 4*mat[0][0]) + k*mat[0][0]*(1 - mat[0][0]**2)
    dtimemat[0][N-1] = D*(mat[1][N-1] - 4*mat[0][N-1] + mat[0][N-2]) + k*mat[0][N-1]*(1 - mat[0][N-1]**2)
    dtimemat[N-1][0] = D*(mat[N-1][1] - 4*mat[N-1][0] + mat[N-2][0]) + k*mat[N-1][0]*(1 - mat[N-1][0]**2)
    dtimemat[N-1][N-1] = D*(-4*mat[N-1][N-1] + mat[N-1][N-2] + mat[N-2][N-1]) + k*mat[N-1][N-1]*(1 - mat[N-1][N-1]**2)
    dtimemat[:,0][1:N-1] = D*(mat[:,1][1:N-1] + mat[:,0][2:N] - 4*mat[:,0][1:N-1] + mat[:,0][0:N-2]) + k*np.multiply(mat[:,0][1:N-1],1-np.square(mat[:,0][1:N-1]))
    dtimemat[0,:][1:N-1] = D*(mat[0,:][2:N] + mat[1,:][1:N-1] - 4*mat[0,:][1:N-1] + mat[0,:][0:N-2]) + k*np.multiply(mat[0,:][1:N-1],1-np.square(mat[0,:][1:N-1]))
    dtimemat[:,N-1][1:N-1] = D*(mat[:,N-1][2:N] -4*mat[:,N-1][1:N-1] + mat[:,N-2][1:N-1] + mat[:,N-1][0:N-2]) + k*np.multiply(mat[:,N-1][1:N-1],1-np.square(mat[:,N-1][1:N-1]))
    dtimemat[N-1,:][1:N-1] = D*(mat[N-1,:][2:N] -4*mat[N-1,:][1:N-1] + mat[N-1,:][0:N-2] + mat[N-2,:][1:N-1]) + k*np.multiply(mat[N-1,:][1:N-1],1-np.square(mat[N-1,:][1:N-1]))
    dtimemat[1:N-1, 1:N-1] = D*(mat[1:N-1, 2:N] + mat[2:N, 1:N-1] -4*mat[1:N-1, 1:N-1] + mat[1:N-1, 0:N-2] + mat[0:N-2, 1:N-1]) + k*np.multiply(mat[1:N-1, 1:N-1],1-np.square(mat[1:N-1, 1:N-1]))
    return dtimemat
  
  # Initiate the first condition
N = 160
grid = np.zeros((N,N))[0]
ic = np.random.random((N, N)) - 0.5
collect_mat = []
collect_mat.append(ic)
D = 1
k = 1
t0 = 0
nt = 2000
dt = 0.02
kk = 0
while kk < nt+1:
    mat = collect_mat[-1]
    dtimemat = dotrho(mat)
    v2mat = mat + dt*dtimemat
    collect_mat.append(v2mat)
    if kk%200 ==0:
        time = t0 + kk*dt
        plt.figure()
        plt.imshow(collect_mat[kk], vmin=-1, vmax=1)
        plt.title('Grid Space at Time={}'.format(time))
        plt.savefig('ca{}.png'.format(kk))
    kk = kk + 1

