#date: 2024-02-14T16:49:56Z
#url: https://api.github.com/gists/b6c34c2033bb7022d19a663bdb586160
#owner: https://api.github.com/users/9kin

import numpy as np
from matplotlib import pyplot as plt

fig, (ax_z, ax_w) = plt.subplots(
    nrows=1,
    ncols=2,
)


# plt.get_current_fig_manager().full_screen_toggle()

POINT_STEP = 5
plt.legend(("f(x)", "f(x * 0.2)"))

ax_w.set_title("$w$-plane")
ax_w.set_xlabel(r"$u$")
ax_w.set_ylabel(r"$v$")
ax_w.set_aspect("equal", "box")
ax_w.grid()
ax_w.scatter(0, 0, color="red", marker="+", label=r"выколотая точка $w = 0$")
ax_w.legend()

ax_z.set_title("$z$-plane")
ax_z.set_xlabel(r"$x$")
ax_z.set_ylabel(r"$y$")
ax_z.set_aspect("equal", "box")
ax_z.grid()
ax_z.scatter(0, 0, color="red", marker="+", label=r"выколотая точка $z = 0$")
ax_z.legend()


def scale():
    rz = [i for i in range(-10, 11, 2)]
    # rw = [i for i in range(-150, 151, 30)]
    rw = [i for i in range(-40, 41, 5)]

    ax_z.set_xlim([min(rz), max(rz)])
    ax_z.set_ylim([min(rz), max(rz)])

    ax_z.set_xticks(rz)
    ax_z.set_yticks(rz)

    ax_w.set_xlim([min(rw), max(rw)])
    ax_w.set_ylim([min(rw), max(rw)])
    ax_w.set_xticks(rw)
    ax_w.set_yticks(rw)

class Paint:
    def __init__(self):
        self.a = []
        fig.canvas.mpl_connect("motion_notify_event", self.motion)
        fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.onclick_flag = False
        self.render()

    def onclick(self, event):
        if event.inaxes != ax_z:
            return
        if not self.onclick_flag:
            self.a.append([[], []])
        self.onclick_flag = not self.onclick_flag

    def mapping(self, z: complex):
        return (z**2)

    def wplane(self):
        for xs, ys in self.a:
            c = [complex(xs[i], ys[i]) for i in range(len(xs))]
            w = list(map(self.mapping, c))
            u = [c.real for c in w]
            v = [c.imag for c in w]
            yield u, v

    def scatter(self, z, w):
        n = 0
        for i in range(len(z)):
            for j in range(0, len(z[i][0]), POINT_STEP):
                xmin, xmax = ax_z.get_xlim()
                ymin, ymax = ax_z.get_ylim()
                x, y = z[i][0][j], z[i][1][j]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax_z.scatter(x, y)
                    ax_z.annotate("$z_{" + str(n + 1) + "}$", (x, y))

                    n += 1

                xmin, xmax = ax_w.get_xlim()
                ymin, ymax = ax_w.get_ylim()
                x, y = w[i][0][j], w[i][1][j]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax_w.scatter(x, y)
                    ax_w.annotate("$w_{" + str(n + 1) + "}$", (x, y))


    def render(self):

        for xs, ys in self.a:
            ax_z.plot(xs, ys)
        wplane = list(self.wplane())
        for xs, ys in wplane:
            ax_w.plot(xs, ys)
        self.scatter(self.a, wplane)
        scale()
        fig.canvas.draw()
    def motion(self, event):
        if event.inaxes != ax_z:
            return
        if self.onclick_flag:
            self.a[-1][0].append(event.xdata)
            self.a[-1][1].append(event.ydata)
            self.render()

p = Paint()
def circle(x0, y0, R, custom_xs=None):
    if custom_xs is not None:
        xs = []
        for x in custom_xs:
            if x0 - R <= x <=  x0 + R:
                xs.append(x)
    else:
        xs = np.linspace(x0 - R, x0 + R, 100)
    ys1 = np.vectorize(lambda x : y0 + (R**2 - (x - x0)**2)**.5)(xs)
    ys2 = np.vectorize(lambda x: y0 - (R ** 2 - (x - x0) ** 2) ** .5)(xs)
    # p.a.append([xs, ys1])
    p.a.append([np.concatenate([xs, xs[::-1]]), np.concatenate([ys1, ys2[::-1]])])

def horizontal(y):
    xs = np.linspace(-15, 15, 100)
    p.a.append([xs, [y for _ in range(len(xs))]])

def vertical(x):
    ys = np.linspace(-15, 15, 100)
    p.a.append([[x for _ in range(len(ys))], ys])


circle(0, 0, 5)
# circle(2, 2, 3)


#horizontal(-5.5)
#horizontal(-2)
#horizontal(0)
#horizontal(2)
#horizontal(3.5)


p.render()


plt.show()