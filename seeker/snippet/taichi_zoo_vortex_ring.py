#date: 2021-08-31T03:20:48Z
#url: https://api.github.com/gists/45de7d54f0cf9bb9183b95454d4ed551
#owner: https://api.github.com/users/k-ye

# C++ reference and tutorial (Chinese): https://zhuanlan.zhihu.com/p/26882619

import taichi as ti

ti.init(arch=ti.gpu)

eps = 0.01
dt = 0.005

n_vortex = 4
n_tracer = 4000
PI = 3.141592653

pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
new_pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
vort = ti.field(ti.f32, shape=n_vortex)

tracer = ti.Vector.field(2, ti.f32, shape=n_tracer)
tracer_vis = ti.Vector.field(2, ti.f32, shape=n_tracer)

@ti.func
def compute_u_single(p, i):
    r2 = (p - pos[i]).norm()**2
    uv = ti.Vector([pos[i].y - p.y, p.x - pos[i].x])
    return vort[i] * uv / (r2 * PI) * 0.5 * (1.0 - ti.exp(-r2 / eps**2))


@ti.func
def compute_u_full(p):
    u = ti.Vector([0.0, 0.0])
    for i in range(n_vortex):
        u += compute_u_single(p, i)
    return u


@ti.kernel
def integrate_vortex():
    for i in range(n_vortex):
        v = ti.Vector([0.0, 0.0])
        for j in range(n_vortex):
            if i != j:
                v += compute_u_single(pos[i], j)
        new_pos[i] = pos[i] + dt * v

    for i in range(n_vortex):
        pos[i] = new_pos[i]


@ti.kernel
def advect():
    for i in range(n_tracer):
        # Ralston's third-order method
        p = tracer[i]
        v1 = compute_u_full(p)
        v2 = compute_u_full(p + v1 * dt * 0.5)
        v3 = compute_u_full(p + v2 * dt * 0.75)
        tracer[i] += (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3) * dt


@ti.kernel
def update_tracer_vis():
    for i in tracer:
        tracer_vis[i] = tracer[i] * ti.Vector([0.05, 0.1]) + ti.Vector([0.0, 0.5])

@ti.kernel
def init_params():
    pos[0] = [0, 1]
    pos[1] = [0, -1]
    pos[2] = [0, 0.3]
    pos[3] = [0, -0.3]
    vort[0] = 1
    vort[1] = -1
    vort[2] = 1
    vort[3] = -1


@ti.kernel
def init_tracers():
    for i in range(n_tracer):
        tracer[i] = [ti.random() - 0.5, ti.random() * 3 - 1.5]


init_params()
init_tracers()

gui = ti.GUI("Vortex Rings", (512, 256), background_color=0xFFFFFF)


while True:
    for i in range(4):  # substeps
        advect()
        integrate_vortex()

    gui.clear(0xFFFFFF)
    update_tracer_vis()
    gui.circles(tracer_vis.to_numpy(),
                radius=1,
                color=0x010101)

    gui.show()
