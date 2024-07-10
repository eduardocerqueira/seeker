#date: 2024-07-10T16:48:06Z
#url: https://api.github.com/gists/d7b45929fd4420e1a1ae41ecead98e50
#owner: https://api.github.com/users/garrettdreyfus

import numpy as np
import matplotlib.pyplot as plt
from ocean_wave_tracing.ocean_wave_tracing import Wave_tracing
from scipy.ndimage.filters import uniform_filter
nx = 100; ny = 100 # number of grid points in x- and y-direction
nb_wave_rays = 100
depth = np.ones((nx,ny))  
x = np.linspace(0,1000,nx) # size x-domain [m]
y = np.linspace(0,1000,ny) # size y-domain [m]
X,Y = np.meshgrid(x,y)
depth = -(1000-np.sqrt((X-1000)**2+Y**2))
## THIS IS WHERE I MAKE THE SAND BAR
depth[30:50,50]=depth[30:50,50]+500
# have to smooth it out or things get too crazy
depth = uniform_filter(depth,4)
depth[depth>=0]=np.nan


T = 250
wt = Wave_tracing(U=np.zeros((ny,nx)),V=np.zeros((ny,nx)),
    nx=nx, ny=ny, nt=150,T=T,
    dx=x[1]-x[0],dy=y[1]-y[0],
    nb_wave_rays=nb_wave_rays,
    domain_X0=x[0], domain_XN=x[-1],
    domain_Y0=y[0], domain_YN=y[-1],
    d=depth)
# Set initial conditions
wt.set_initial_condition(wave_period=20,
    theta0=np.linspace(0.5*np.pi,1*np.pi,nb_wave_rays)[::-1],
    ipx=np.linspace(950,1000,nb_wave_rays),
    ipy=np.linspace(0,50,nb_wave_rays)
)
# Solve
wt.solve()
xx,yy,hm = wt.ray_density(10,10)
print(xx.shape)

fig, (ax1,ax2) = plt.subplots(1,2);
pc=ax2.pcolormesh(wt.x,wt.y,depth,shading='auto');
fig.colorbar(pc,ax=ax2)

ax1.pcolormesh(xx,yy,hm)

for ray_id in range(wt.nb_wave_rays):
    ax2.plot(wt.ray_x[ray_id,:],wt.ray_y[ray_id,:],'-k')

ax2.set_xlim(np.min(wt.x),np.max(wt.x))
ax2.set_ylim(np.min(wt.y),np.max(wt.y))
plt.show()
