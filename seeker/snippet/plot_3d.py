#date: 2021-11-01T17:05:57Z
#url: https://api.github.com/gists/8d6dcc1cdf37f381655f4b61ab256044
#owner: https://api.github.com/users/KrisYu

# plot 3d points

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

# points np array
ax.scatter(points[:,0], points[:,1], points[:,2])


plt.show()

