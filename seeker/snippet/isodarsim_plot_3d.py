#date: 2022-07-11T17:01:50Z
#url: https://api.github.com/gists/86ae10e00bbdc2277860ae4c6989ee35
#owner: https://api.github.com/users/BenSmithers

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from math import pi

import numpy as np

data = np.loadtxt("IsoDAR60MeV-trackOrbit.dat",dtype=str,comments="#")

# use column 1 to make a mask depending on particle ID 
id0 = np.transpose(data[np.transpose(data)[0]=="ID0"])
id1 = np.transpose(data[np.transpose(data)[0]=="ID1"])

# Grab the beta_x, beta_y, beta_z for each of these subsets 
id0_x = id0[2].astype(float)
id0_y = id0[4].astype(float)
id0_z = id0[6].astype(float)
id1_x = id0[2].astype(float)
id1_y = id0[4].astype(float)
id1_z = id0[6].astype(float)


# plot everything! 
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(id0_x,id0_y,id0_z, label='ID0')
ax.plot3D(id1_x,id1_y,id1_z, label='ID1')

ax.set_zlim([-0.5,0.5])
ax.set_xlabel("X",size=14)
ax.set_ylabel("Y",size=14)
ax.set_zlabel("Z",size=14)
ax.legend()

plt.show()
