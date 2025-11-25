#date: 2025-11-25T16:54:37Z
#url: https://api.github.com/gists/e2bf72ea52eb7a57558d7802aedc237d
#owner: https://api.github.com/users/worldmaker18349276

# interactive_curvature.py
import numpy as np
import matplotlib.pyplot as plt

# initial points
pts = np.array([[0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.5, 1.0],
                ])

t = np.linspace(-1, 2, 301)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-3, 4)
ax.set_ylim(-2, 3)
ax.set_title("Drag points; curvature at middle point shown\nkappa = |v x a| / |v|^3")

scatter = ax.scatter(pts[:,0], pts[:,1], s=100, zorder=5, picker=True)

# arrows for gradients (one arrow per point)
grad_quiver = ax.quiver(pts[:,0], pts[:,1], np.zeros(pts.shape[0]), np.zeros(pts.shape[0]), angles='xy', scale_units='xy', scale=1, width=0.007, zorder=4)
# arrows for v and a at middle point
vec_quiver = ax.quiver([pts[1,0], pts[1,0]], [pts[1,1], pts[1,1]], [0,0], [0,0], angles='xy', scale_units='xy', scale=1, width=0.01, color=['tab:green','tab:orange'], zorder=3)
kappa_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

picked_idx = None
dragging = False
grad_scale_factor = 1.0     # user-adjustable gradient scale

def curve(p):
    if p.shape[0] == 3:
        # p: array shape (3,2)
        p0, p1, p2 = p
        p3 = p2 + (p2 - p1)

    if p.shape[0] == 4:
        # p: array shape (4,2)
        p0, p1, p2, p3 = p

    b0 = (-t**3 + 3 * t**2 - 3 * t + 1) / 6
    b1 = (3 * t**3 - 6 * t**2 + 4) / 6
    b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6
    b3 = (t**3) / 6
    
    return b0[:,None] * p0[None,:] + b1[:,None] * p1[None,:] + b2[:,None] * p2[None,:] + b3[:,None] * p3[None,:]

c = curve(pts)
curv_plot = ax.plot(c[0:101,0], c[0:101,1], color="green", zorder=2)
curv_plot += ax.plot(c[100:201,0], c[100:201,1], color="orange", zorder=2)
curv_plot += ax.plot(c[200:301,0], c[200:301,1], color="green", zorder=2)

def curvature(p):
    if p.shape[0] == 3:
        # p: array shape (3,2)
        p0, p1, p2 = p
        center = (p0 + 4*p1 + p2) / 6.0
        # treat times t = -1, 0, 1 -> central finite diff
        v = (p2 - p0) / 2.0  # approximate velocity at middle
        a = p2 - 2*p1 + p0  # approximate acceleration (dt=1)
        cross = v[0]*a[1] - v[1]*a[0]  # scalar z-component of cross product in 2D
        vnorm = np.linalg.norm(v)
        if vnorm == 0:
            return 0.0, center, v, a, cross
        kappa = abs(cross) / (vnorm**3)
        return kappa, center, v, a, cross

    if p.shape[0] == 4:
        # p: array shape (4,2)
        p0, p1, p2, p3 = p
        center = (p0 + 23*p1 + 23*p2 + p3) / 48.0
        v = (p3 + 5*p2 - 5*p1 - p0) / 8.0
        a = (p3 - p2 - p1 + p0) / 2.0
        cross = v[0]*a[1] - v[1]*a[0]  # scalar z-component of cross product in 2D
        vnorm = np.linalg.norm(v)
        if vnorm == 0:
            return 0.0, center, v, a, cross
        kappa = abs(cross) / (vnorm**3)
        return kappa, center, v, a, cross

def numeric_gradient_kappa(p, eps=1e-6):
    # returns gradient vectors (3,2) of kappa wrt each point coordinate
    base_kappa, *_ = curvature(p)
    grads = np.zeros_like(p)
    for i in range(p.shape[0]):
        for d in range(p.shape[1]):
            dp = np.zeros_like(p)
            dp[i,d] = eps
            kp = curvature(p + dp)[0]
            km = curvature(p - dp)[0]
            grads[i,d] = (kp - km) / (2*eps)
    return grads

def update_plot():
    global grad_quiver, vec_quiver, curv_plot
    kappa, center, v, a, cross = curvature(pts)
    grads = numeric_gradient_kappa(pts, eps=1e-6)
    # Update scatter
    scatter.set_offsets(pts)
    # Update gradient quiver
    grad_quiver.remove()

    grad_quiver = ax.quiver(
        pts[:,0], pts[:,1],
        grads[:,0]*grad_scale_factor, grads[:,1]*grad_scale_factor,
        angles='xy', scale_units='xy', scale=1,
        width=0.007, color='tab:blue', zorder=4
    )

    # Update v and a arrows at middle point
    vec_quiver.remove()
    # scale v and a for plotting (not mathematical scale)
    vscale = 0.7 / (np.linalg.norm(v) + 1e-9)
    ascale = 0.7 / (np.linalg.norm(a) + 1e-9)
    vec_quiver = ax.quiver([center[0], center[0]], [center[1], center[1]], [v[0]*vscale, a[0]*ascale], [v[1]*vscale, a[1]*ascale],
                           angles='xy', scale_units='xy', scale=1, width=0.01, color=['tab:green','tab:orange'], zorder=3)

    for curv_plot_ in curv_plot:
        curv_plot_.remove()
    c = curve(pts)
    curv_plot = ax.plot(c[0:101,0], c[0:101,1], color="green", zorder=2)
    curv_plot += ax.plot(c[100:201,0], c[100:201,1], color="orange", zorder=2)
    curv_plot += ax.plot(c[200:301,0], c[200:301,1], color="green", zorder=2)

    # text
    kappa_text.set_text(f"curvature kappa = {kappa:.6g}\n|v| = {np.linalg.norm(v):.6g}, |a| = {np.linalg.norm(a):.6g}\ncross = {cross:.6g}")
    fig.canvas.draw_idle()

def on_press(event):
    global picked_idx, dragging
    if event.inaxes is not ax:
        return
    # find nearest point
    mouse = np.array([event.xdata, event.ydata])
    dists = np.linalg.norm(pts - mouse, axis=1)
    idx = np.argmin(dists)
    if dists[idx] < 0.15:  # pick threshold (in data units)
        picked_idx = idx
        dragging = True

def on_motion(event):
    global picked_idx, dragging
    if not dragging or picked_idx is None:
        return
    if event.inaxes is not ax:
        return
    # update picked point
    pts[picked_idx, 0] = event.xdata
    pts[picked_idx, 1] = event.ydata
    update_plot()

def on_release(event):
    global picked_idx, dragging
    picked_idx = None
    dragging = False

def on_scroll(event):
    global grad_scale_factor
    # event.step: +1 (scroll up) or -1 (scroll down)
    if event.step > 0:
        grad_scale_factor *= 1.1
    else:
        grad_scale_factor *= 0.9
    update_plot()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)

# initial draw
update_plot()
plt.show()

