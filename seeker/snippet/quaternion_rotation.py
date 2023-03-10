#date: 2023-03-10T16:40:30Z
#url: https://api.github.com/gists/c445953866b10ebb29a1676dbb717135
#owner: https://api.github.com/users/dsuhoi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

N = 3
coord = np.array([[int(b) for b in f"{i:0{N}b}"] for i in range(2**N)], dtype="f4")


class Quat:
    def __init__(self, a, b, c, d):
        self._p = [a, b, c, d]

    def __mul__(self, q):
        (a, b, c, d), (a_, b_, c_, d_) = self._p, q._p
        return Quat(
            a * a_ - b * b_ - c * c_ - d * d_,
            a * b_ + b * a_ + c * d_ - d * c_,
            a * c_ - b * d_ + c * a_ + d * b_,
            a * d_ + b * c_ - c * b_ + d * a_,
        )

    def T(self):
        return Quat(self._p[0], -self._p[1], -self._p[2], -self._p[3])


angle = np.pi / 50
vect = [1, 1, 2]
vect = np.array(vect) / np.linalg.norm(vect)
r = Quat(np.cos(angle / 2), *(np.sin(angle / 2) * vect).tolist())

fig = plt.figure()
lim = [-0.5, 1.5]
ax = fig.add_subplot(projection="3d", xlim=lim, ylim=lim, zlim=lim)
(cube,) = ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color="k")


def rotate(t):
    coord[:] = [(r * Quat(0, *coord[i, :].tolist()) * r.T())._p[1:] for i in range(8)]
    cube.set_data(coord[:, 0], coord[:, 1])
    cube.set_3d_properties(coord[:, 2])


anim = FuncAnimation(fig, rotate, interval=1000 / 30, cache_frame_data=False)
plt.show()