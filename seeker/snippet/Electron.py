#date: 2022-10-07T17:18:08Z
#url: https://api.github.com/gists/85553ebb50d0323a75efae6b0af68692
#owner: https://api.github.com/users/MBobkov

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro')
ln1, = plt.plot([], [], 'ro')
plt.title('Electron and nucleus')
plt.xlabel('Axis X')
plt.ylabel('Axis Y')
circle2 = plt.Circle((0, 0), 5, fill=False)
circle3 = plt.Circle((0, 0), 2.5, fill=False)


def circle(t, r=5, r1=10, w=10, w1=20):
    x = r*np.sin(w*t)
    y = r*np.cos(w*t)
    return x, y


def circle1(t, r=2.5, r1=10, w=20, w1=20):
    x = r*np.sin(w*t)
    y = r*np.cos(w*t)
    return x, y


def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    #return ln, ln1


def animation(frame):
    ln.set_data(circle(frame)[0], circle(frame)[1])
    ln1.set_data(circle1(frame)[0], circle1(frame)[1])
    #return ln, ln1


ani = FuncAnimation(fig, animation, frames=np.linspace(1, 10, 1000), init_func=init, interval=0.05)
plt.scatter(0, 0, s=70, c='blue')
ax.add_patch(circle2)
ax.add_patch(circle3)
plt.show()