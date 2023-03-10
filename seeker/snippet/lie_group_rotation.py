#date: 2023-03-10T16:40:30Z
#url: https://api.github.com/gists/c445953866b10ebb29a1676dbb717135
#owner: https://api.github.com/users/dsuhoi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

N = 3
coord = np.array([[int(b) for b in f"{i:0{N}b}"] for i in range(2**N)], dtype="f4")

ang_x = np.pi / 80
ang_y = np.pi / 50
ang_z = np.pi / 40

Lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype="f4")
Ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype="f4")
Lz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype="f4")
R = expm(ang_x * Lx) @ expm(ang_y * Ly) @ expm(ang_z * Lz)

fig = plt.figure()
lim = [-0.5, 1.5]
ax = fig.add_subplot(projection="3d", xlim=lim, ylim=lim, zlim=lim)
(cube,) = ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color="k")


def rotate(t):
    coord[:] = [R @ coord[i, :] for i in range(8)]
    cube.set_data(coord[:, 0], coord[:, 1])
    cube.set_3d_properties(coord[:, 2])


anim = FuncAnimation(fig, rotate, interval=1000 / 30, cache_frame_data=False)
plt.show()