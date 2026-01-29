#date: 2026-01-29T17:08:36Z
#url: https://api.github.com/gists/157592b4c4db5b93bafe4b4df4ded7f7
#owner: https://api.github.com/users/zhouanqi1231

import pygame
import pygame.freetype
import math
import random

"""
3x2 SVD Visualizer with 3D-style view + thermometer calibration.

World axes (right-handed):
  X: forward (into the screen)  -> parameter a
  Y: right                      -> parameter b
  Z: up

We draw:
  - ab-plane (Z=0) as a tilted grid "floor"
  - Z axis vertical in the window
  - Unit circle in the ab-plane (parameter space: a,b)
  - M·circle (3D ellipse) lifted above the floor, with vertical drop lines
  - Ground-truth point g in 3D
  - Fitted point s_hat = M β on the ellipse
  - Residual vector r = s - s_hat

M is 3x2: M : R^2 -> R^3
SVD: M = U Σ V^T, with
  U: 3x3
  Σ: 3x2 (σ1, σ2) on diagonal
  V: 2x2

Thermometer calibration:
  Data matrix M (3x2) built from sensor readings
  Ground-truth vector g = (G1, G2, G3)^T
  β = (a, b) = V Σ^+ U^T g
"""

# =========================
# Window & drawing settings
# =========================
WINSIZE = (900, 900)
BG = (0, 0, 0)
GRID = (60, 60, 60)

CIRCLE_COLOR = (180, 180, 180)  # original unit circle (input plane Z=0)
BASIS_COLOR = (180, 180, 180)  # original basis in input plane

U_COLOR = (255, 100, 200)  # unified color for final ellipse (M-only and SVD final)

V_COLOR = (100, 200, 255)  # after V^T (still in plane Z=0)
SIGMA_COLOR = (100, 255, 150)  # after Σ V^T (ellipse in plane Z=0)

V_BASIS_COLOR = (100, 200, 255)
SIGMA_BASIS_COLOR = (100, 255, 150)
U_BASIS_COLOR = (255, 100, 200)

LIFT_LINE_COLOR = (120, 120, 120)  # vertical projection lines from ellipse to ab-plane

GROUND_TRUTH_COLOR = (255, 255, 0)  # point g (star)
FIT_POINT_COLOR = U_COLOR  # point Mβ (star) – magenta like the final ellipse
BETA_COLOR = (200, 200, 200)  # β in (a,b)-plane (star)
RESIDUAL_COLOR = (255, 255, 255)  # residual arrow

CENTER = (WINSIZE[0] // 2, WINSIZE[1] // 2)
SCALE = 160.0  # pixels per world unit

# --------- global state ---------

# Quaternion for U rotation (precomputed from U_current)
qU_target = (1.0, 0.0, 0.0, 0.0)

# M is 3x2: rows: [m11 m12], [m21 m22], [m31 m32]
M_current = [[1.0, 0.3], [0.1, 0.8], [0.0, 0.0]]

U_current = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

V_current = [[1.0, 0.0], [0.0, 1.0]]

sing_vals = (1.0, 1.0)  # (σ1, σ2)

draw_mode = "M_only"  # "M_only" or "SVD_steps"

# Animation state for SVD steps
anim_running = False
# anim_phase: 0 = idle/finished; 1 = V^T; 2 = Σ; 3 = U
anim_phase = 0
anim_frame = 0
PHASE_FRAMES = 40

# Animation state for M-only morph (unit circle -> M·circle)
a_only_t = 0.0  # interpolation parameter in [0,1]

# Ground truth and calibration
s_vector = (0.0, 1.0, 2.0)  # S1,S2,S3
theta_hat = None  # (a, b) representing β
s_hat_vector = None  # M β
calib_valid = False  # flag if β is defined & computed

# ======== Pygame init ========
pygame.init()
pygame.freetype.init()
screen = pygame.display.set_mode(WINSIZE)
pygame.display.set_caption("3x2 SVD & Thermometer Calibration (3D view)")
screen.fill(BG)
pygame.display.flip()

LABEL_FONT = pygame.freetype.SysFont("FreeSerif", 22)
STATUS_FONT = pygame.freetype.SysFont("FreeSerif", 20)
LEGEND_FONT = pygame.freetype.SysFont("FreeSans", 18)
MATRIX_FONT = pygame.freetype.SysFont("Courier New", 14)

running = True

# =====================  Basic linalg helpers (2D & 3D)  =====================

# --- 2D vector ops (input / parameter space) ---


def vec2_add(u, v):
    return (u[0] + v[0], u[1] + v[1])


def vec2_sub(u, v):
    return (u[0] - v[0], u[1] - v[1])


def vec2_scale(s, v):
    return (s * v[0], s * v[1])


def vec2_dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def vec2_norm(v):
    return math.sqrt(vec2_dot(v, v))


def vec2_normalize(v):
    n = vec2_norm(v)
    if n < 1e-12:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def vec2_perp(v):
    return (-v[1], v[0])


# --- 3D vector ops (output space) ---


def vec3_add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def vec3_sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def vec3_scale(s, v):
    return (s * v[0], s * v[1], s * v[2])


def vec3_dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def vec3_norm(v):
    return math.sqrt(vec3_dot(v, v))


def vec3_normalize(v):
    n = vec3_norm(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def vec3_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# --- matrices ---


def mat2_mul(M, N):
    """2x2 * 2x2"""
    return [
        [M[0][0] * N[0][0] + M[0][1] * N[1][0], M[0][0] * N[0][1] + M[0][1] * N[1][1]],
        [M[1][0] * N[0][0] + M[1][1] * N[1][0], M[1][0] * N[0][1] + M[1][1] * N[1][1]],
    ]


def mat2_vec2(M, v):
    """2x2 * 2D vector"""
    return (M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1])


def mat3_vec3(M, v):
    """3x3 * 3D vector"""
    return (
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    )


def mat3_det(M):
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])


def mat3_mat3(M, N):
    """3x3 * 3x3"""
    res = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            res[i][j] = M[i][0] * N[0][j] + M[i][1] * N[1][j] + M[i][2] * N[2][j]
    return res


def mat3_from_axis_angle(axis, theta):
    """Rodrigues' formula for 3D rotation."""
    ax = vec3_normalize(axis)
    x, y, z = ax
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1.0 - c
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


# =====================  Quaternion helpers  =====================


def quat_from_axis_angle(axis, theta):
    axis = vec3_normalize(axis)
    half = theta * 0.5
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_to_matrix(q):
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return [[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]]


def quat_from_rotation_matrix(M):
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z)."""
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21, m22 = M[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    # Normalize to be safe
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def quat_slerp(q0, q1, t):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1

    # Ensure shortest path
    if dot < 0.0:
        dot = -dot
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1

    if dot > 0.9995:
        # Very close: linear interpolate and normalize
        w = w0 + t * (w1 - w0)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return (w / norm, x / norm, y / norm, z / norm)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-8:
        return q0

    w0_factor = math.sin((1.0 - t) * theta) / sin_theta
    w1_factor = math.sin(t * theta) / sin_theta

    w = w0_factor * w0 + w1_factor * w1
    x = w0_factor * x0 + w1_factor * x1
    y = w0_factor * y0 + w1_factor * y1
    z = w0_factor * z0 + w1_factor * z1
    return (w, x, y, z)


# =====================  3x2 SVD (analytic)  =====================


def eigenvector_sym_2x2(a11, a12, a22, lam):
    """
    Eigenvector of symmetric 2x2 [[a11, a12], [a12, a22]] for eigenvalue lam.
    Returns non-normalized vector (vx, vy).
    """

    # --- IMPORTANT FIX ---
    # If the matrix is (nearly) diagonal, the eigenvectors are just the axes.
    # Choose the axis corresponding to the closer diagonal entry.
    if abs(a12) < 1e-10:
        if abs(lam - a11) <= abs(lam - a22):
            return (1.0, 0.0)  # eigenvector for a11
        else:
            return (0.0, 1.0)  # eigenvector for a22

    # Generic (off-diagonal) case: solve (A - lam I)v = 0
    vx = a12
    vy = lam - a11

    # If that’s tiny (rare numerical corner), use the other row equation
    if abs(vx) + abs(vy) < 1e-12:
        vx = lam - a22
        vy = a12

    return (vx, vy)


def svd_3x2(M):
    """
    SVD of a 3x2 matrix M.
    M = U Σ V^T
      U: 3x3 orthogonal, det(U) = +1
      Σ: 3x2 with (σ1, σ2) on diag, σi ≥ 0
      V: 2x2 orthogonal, det(V) = +1
    """
    m11, m12 = M[0]
    m21, m22 = M[1]
    m31, m32 = M[2]

    c1 = (m11, m21, m31)
    c2 = (m12, m22, m32)

    # M^T M (2x2)
    ata11 = vec3_dot(c1, c1)
    ata12 = vec3_dot(c1, c2)
    ata22 = vec3_dot(c2, c2)

    trace = ata11 + ata22
    det = ata11 * ata22 - ata12 * ata12
    disc = trace * trace / 4.0 - det
    if disc < 0.0:
        disc = 0.0
    root = math.sqrt(disc)
    lam1 = trace / 2.0 + root
    lam2 = trace / 2.0 - root

    if lam2 > lam1:
        lam1, lam2 = lam2, lam1

    s1 = math.sqrt(lam1) if lam1 > 0.0 else 0.0
    s2 = math.sqrt(lam2) if lam2 > 0.0 else 0.0

    # ----- Right singular vectors (2D) -----
    v1 = eigenvector_sym_2x2(ata11, ata12, ata22, lam1)
    v1 = vec2_normalize(v1)
    if vec2_norm(v1) < 1e-12:
        v1 = (1.0, 0.0)

    v2 = eigenvector_sym_2x2(ata11, ata12, ata22, lam2)
    proj = vec2_scale(vec2_dot(v2, v1), v1)
    v2 = vec2_sub(v2, proj)
    if vec2_norm(v2) < 1e-8:
        v2 = vec2_perp(v1)
    v2 = vec2_normalize(v2)

    # Enforce det(V) = +1 (right-handed 2D basis)
    detV = v1[0] * v2[1] - v1[1] * v2[0]
    if detV < 0.0:
        # Flip second singular direction; Σ stays ≥0, we’ll absorb the sign in U
        v2 = (-v2[0], -v2[1])

    V = [[v1[0], v2[0]], [v1[1], v2[1]]]

    # ----- Left singular vectors u_i = (1/σ_i) M v_i -----
    def compute_u_from_v_and_sigma(v, s):
        if s < 1e-8:
            return (0.0, 0.0, 0.0)
        x, y = v
        return ((m11 * x + m12 * y) / s, (m21 * x + m22 * y) / s, (m31 * x + m32 * y) / s)

    u1 = compute_u_from_v_and_sigma(v1, s1)
    u2 = compute_u_from_v_and_sigma(v2, s2)

    u1 = vec3_normalize(u1)
    proj_u2_u1 = vec3_scale(vec3_dot(u2, u1), u1)
    u2 = vec3_sub(u2, proj_u2_u1)
    if vec3_norm(u2) < 1e-8:
        if abs(u1[0]) < 0.9:
            u2 = vec3_normalize(vec3_cross(u1, (1.0, 0.0, 0.0)))
        else:
            u2 = vec3_normalize(vec3_cross(u1, (0.0, 1.0, 0.0)))
    else:
        u2 = vec3_normalize(u2)

    # Third column: complete to an orthonormal right-handed basis
    u3 = vec3_cross(u1, u2)
    u3 = vec3_normalize(u3)

    U = [
        [u1[0], u2[0], u3[0]],
        [u1[1], u2[1], u3[1]],
        [u1[2], u2[2], u3[2]],
    ]

    # Just in case numerical issues made it left-handed, flip u3.
    if mat3_det(U) < 0.0:
        U[0][2] = -U[0][2]
        U[1][2] = -U[1][2]
        U[2][2] = -U[2][2]

    return U, (s1, s2), V


# =====================  3D projection & drawing helpers  =====================


def project_3d_to_2d(x, y, z):
    """
    Oblique-ish projection.

    World (right-handed):
      X: forward (into screen) -> a
      Y: right                 -> b
      Z: up

    We project so:
      - Y is mostly horizontal
      - Z is vertical
      - X (depth) gives a small shift left/down to suggest depth
        (positive X now leans down-left on screen)
    """
    xp = y - 0.4 * x
    yp = z - 0.4 * x
    return xp, yp


def world_to_screen_3d(x, y, z):
    xp, yp = project_3d_to_2d(x, y, z)
    px = CENTER[0] + int(SCALE * xp)
    py = CENTER[1] - int(SCALE * yp)
    return px, py


def draw_grid_and_axes(surface):
    surface.fill(BG)

    # ab-plane grid at Z=0
    grid_extent = 3
    for gx in range(-grid_extent, grid_extent + 1):
        p1 = world_to_screen_3d(gx, -grid_extent, 0.0)
        p2 = world_to_screen_3d(gx, grid_extent, 0.0)
        pygame.draw.line(surface, GRID, p1, p2, 1)

    for gy in range(-grid_extent, grid_extent + 1):
        p1 = world_to_screen_3d(-grid_extent, gy, 0.0)
        p2 = world_to_screen_3d(grid_extent, gy, 0.0)
        pygame.draw.line(surface, GRID, p1, p2, 1)

    origin = world_to_screen_3d(0.0, 0.0, 0.0)

    # a axis (forward, former X) – appears diagonally down-left
    a_end = world_to_screen_3d(2.0, 0.0, 0.0)
    pygame.draw.line(surface, (220, 80, 80), origin, a_end, 2)

    # b axis (right, former Y) – horizontal
    b_end = world_to_screen_3d(0.0, 2.0, 0.0)
    pygame.draw.line(surface, (80, 220, 80), origin, b_end, 2)

    # Z axis (up) – vertical
    z_end = world_to_screen_3d(0.0, 0.0, 2.0)
    pygame.draw.line(surface, (80, 80, 240), origin, z_end, 2)

    def draw_arrowhead(start, end, color):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        hx, hy = -uy, ux
        head_len = 10
        head_wid = 6
        p1 = end
        p2 = (end[0] - head_len * ux + head_wid * hx, end[1] - head_len * uy + head_wid * hy)
        p3 = (end[0] - head_len * ux - head_wid * hx, end[1] - head_len * uy - head_wid * hy)
        pygame.draw.polygon(surface, color, [p1, p2, p3])

    draw_arrowhead(origin, a_end, (220, 80, 80))
    draw_arrowhead(origin, b_end, (80, 220, 80))
    draw_arrowhead(origin, z_end, (80, 80, 240))

    # Axis labels: a, b, Z
    LABEL_FONT.render_to(surface, (a_end[0] + 4, a_end[1]), "a", (220, 80, 80))
    LABEL_FONT.render_to(surface, (b_end[0] + 4, b_end[1]), "b", (80, 220, 80))
    LABEL_FONT.render_to(surface, (z_end[0] + 4, z_end[1] - 16), "Z", (80, 80, 240))


def generate_unit_circle_2d(n_points=200):
    pts = []
    for k in range(n_points + 1):
        theta = 2 * math.pi * k / n_points
        pts.append((math.cos(theta), math.sin(theta)))
    return pts


def embed_2d_to_3d(p2):
    # Input circle lives in ab-plane: (a, b, Z=0)
    return (p2[0], p2[1], 0.0)


def draw_polyline_3d(surface, pts3, color, width=2, closed=True):
    if len(pts3) < 2:
        return
    pix_pts = [world_to_screen_3d(p[0], p[1], p[2]) for p in pts3]
    if closed:
        pygame.draw.polygon(surface, color, pix_pts, width)
    else:
        pygame.draw.lines(surface, color, False, pix_pts, width)


def draw_arrow_3d(surface, origin3, vec3, color, width=3):
    ox, oy, oz = origin3
    vx, vy, vz = vec3
    start = world_to_screen_3d(ox, oy, oz)
    end = world_to_screen_3d(ox + vx, oy + vy, oz + vz)
    pygame.draw.line(surface, color, start, end, width)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    L = math.hypot(dx, dy)
    if L < 10:
        return
    ux, uy = dx / L, dy / L
    hx, hy = -uy, ux
    head_len = 12
    head_wid = 7
    p1 = (end[0], end[1])
    p2 = (end[0] - head_len * ux + head_wid * hx, end[1] - head_len * uy + head_wid * hy)
    p3 = (end[0] - head_len * ux - head_wid * hx, end[1] - head_len * uy - head_wid * hy)
    pygame.draw.polygon(surface, color, [p1, p2, p3])


def draw_star_3d(surface, p3, color, radius=8, width=2):
    """Draw an asterisk-style star at 3D point."""
    px, py = world_to_screen_3d(p3[0], p3[1], p3[2])
    r = radius
    r_diag = int(radius * 0.7)

    # Horizontal & vertical
    pygame.draw.line(surface, color, (px - r, py), (px + r, py), width)
    pygame.draw.line(surface, color, (px, py - r), (px, py + r), width)
    # Diagonals
    pygame.draw.line(surface, color, (px - r_diag, py - r_diag), (px + r_diag, py + r_diag), width)
    pygame.draw.line(surface, color, (px - r_diag, py + r_diag), (px + r_diag, py - r_diag), width)


def draw_lift_lines(surface, pts3, step=12):
    """
    For every 'step'-th point on a 3D curve, draw a vertical line
    down to the ab-plane (z=0), to visualize lifting.
    """
    for i in range(0, len(pts3), step):
        x, y, z = pts3[i]
        base = (x, y, 0.0)
        p_top = world_to_screen_3d(x, y, z)
        p_base = world_to_screen_3d(base[0], base[1], base[2])
        pygame.draw.line(surface, LIFT_LINE_COLOR, p_base, p_top, 1)


def put_status(surface, text):
    STATUS_FONT.render_to(surface, (20, 20), text, (230, 230, 230))


def put_legend(surface):
    x0, y0 = 20, WINSIZE[1] - 260
    dy = 22

    if draw_mode == "M_only":
        items = [
            ("Unit circle / basis in ab-plane (z=0)", CIRCLE_COLOR),
            ("After UΣVᵀ (ellipse in 3D)", U_COLOR),
            ("Vertical lines: lift from ab floor", LIFT_LINE_COLOR),
            ("Ground truth s (star)", GROUND_TRUTH_COLOR),
            ("Best-fit ŝ = Mβ̂  on ellipse (star)", FIT_POINT_COLOR),
            ("Estimated parameters β̂  in (a,b)-plane (star)", BETA_COLOR),
            ("Residual s - ŝ", RESIDUAL_COLOR),
        ]
    else:
        items = [
            ("Original circle / basis (ab-plane)", CIRCLE_COLOR),
            ("After Vᵀ (ab-plane)", V_COLOR),
            ("After ΣVᵀ (ellipse in ab-plane)", SIGMA_COLOR),
            ("After UΣVᵀ (ellipse in 3D)", U_COLOR),
            ("Vertical lines: lift from ab floor", LIFT_LINE_COLOR),
            ("Ground truth s (star)", GROUND_TRUTH_COLOR),
            ("Best-fit ŝ = Mβ̂  on ellipse (star)", FIT_POINT_COLOR),
            ("Estimated parameters β̂  in (a,b)-plane (star)", BETA_COLOR),
            ("Residual s - ŝ", RESIDUAL_COLOR),
        ]

    for i, (label, col) in enumerate(items):
        y = y0 + i * dy
        pygame.draw.line(surface, col, (x0, y), (x0 + 30, y), 4)
        LEGEND_FONT.render_to(surface, (x0 + 40, y - 8), label, (230, 230, 230))


# =====================  Labels and matrix formatting =====================


def draw_hud(surface):
    # 1. Singular Values
    s1, s2 = sing_vals
    if abs(s2) < 1e-12:
        cond = float("inf")
    else:
        cond = s1 / s2
    sv_text = f"σ₁ = {s1:.3g},  σ₂ = {s2:.3g},  cond(M) ≈ {cond:.3g}"
    STATUS_FONT.render_to(surface, (20, 20), sv_text, (200, 255, 200))

    # 2. Phase Info
    phase_text = ""
    if draw_mode == "M_only":
        phase_text = "Current step: M-only interpolation (unit circle → M·circle)"
    else:
        if anim_phase == 0:
            phase_text = "Current step: all SVD stages visible (Vᵀ, Σ, U)" if not anim_running else "Current step: starting..."
        elif anim_phase == 1:
            phase_text = "Current step: applying Vᵀ (2D rotation/flip)"
        elif anim_phase == 2:
            phase_text = "Current step: applying Σ (stretch in ab-plane)"
        elif anim_phase == 3:
            phase_text = "Current step: applying U (3D rotation of ellipse)"
    STATUS_FONT.render_to(surface, (20, 50), phase_text, (150, 200, 255))

    # 3. Calibration Info
    calib_text = ""
    if not calib_valid or theta_hat is None or s_hat_vector is None:
        calib_text = "Calibration: (ill-conditioned or undefined)"
    else:
        a, b = theta_hat
        r = vec3_sub(s_vector, s_hat_vector)
        rnorm = vec3_norm(r)
        calib_text = f"Calibration β̂ = (A, B) = ({a:.4g}, {b:.4g}),  ||g - Mβ̂|| ≈ {rnorm:.4g}"
    STATUS_FONT.render_to(surface, (20, 80), calib_text, (255, 100, 200))

    # 4. Matrices (simplified, just M and singular values)
    x_mat, y_mat = 20, 120

    def draw_mat_block(label, mat_str):
        nonlocal y_mat
        lines = [label] + mat_str.split("\n")
        for line in lines:
            MATRIX_FONT.render_to(surface, (x_mat, y_mat), line, (255, 255, 255))
            y_mat += 16
        y_mat += 10

    draw_mat_block("M =", format_matrix_3x2(M_current))
    draw_mat_block("U =", format_matrix_3x3(U_current))
    draw_mat_block("Σ =", format_matrix_3x2([[s1, 0.0], [0.0, s2], [0.0, 0.0]]))
    draw_mat_block("Vᵀ =", format_matrix_2x2([[V_current[0][0], V_current[1][0]], [V_current[0][1], V_current[1][1]]]))

    # 5. Controls
    controls = ["Controls:", "[SPACE] Play/Pause Animation", "[R]     Reset Animation", "[M]     Toggle Mode (M-only / SVD steps)", "[G]     Random Well-Conditioned M (good)", "[B]     Random Ill-Conditioned M (bad)", "[S]     Random Ground Truth s", "[ESC]   Quit"]
    y_ctrl = WINSIZE[1] - 160
    x_ctrl = WINSIZE[0] - 350  # Adjust position
    for line in controls:
        STATUS_FONT.render_to(surface, (x_ctrl, y_ctrl), line, (255, 255, 0))
        y_ctrl += 22


# =====================  Axis-angle from orthogonal U =====================


def axis_angle_from_orthogonal(R):
    trace = R[0][0] + R[1][1] + R[2][2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-6:
        return (1.0, 0.0, 0.0), 0.0

    denom = 2.0 * math.sin(theta)
    kx = (R[2][1] - R[1][2]) / denom
    ky = (R[0][2] - R[2][0]) / denom
    kz = (R[1][0] - R[0][1]) / denom
    axis = (kx, ky, kz)
    axis = vec3_normalize(axis)
    return axis, theta


def decompose_orthogonal_3x3(O):
    detO = mat3_det(O)
    if detO >= 0:
        R = O
        F = None
    else:
        F = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
        R = mat3_mat3(O, F)
    axis, theta = axis_angle_from_orthogonal(R)
    return axis, theta, F


# =====================  Animated SVD steps  =====================


def animate_V_step_2d(circle0_2d, e1_2d, e2_2d, t):
    """Animate V^T as 2D rotation+optional reflection in ab-plane."""
    Vt = [[V_current[0][0], V_current[1][0]], [V_current[0][1], V_current[1][1]]]
    detVt = Vt[0][0] * Vt[1][1] - Vt[0][1] * Vt[1][0]
    if detVt >= 0:
        R = Vt
        F = None
    else:
        F = [[1.0, 0.0], [0.0, -1.0]]
        R = mat2_mul(Vt, F)

    angle = math.atan2(R[1][0], R[0][0])
    theta = angle * t
    c = math.cos(theta)
    s = math.sin(theta)
    R_t = [[c, -s], [s, c]]

    def apply_stage(p):
        x, y = p
        if F is not None:
            x, y = x, -y
        return mat2_vec2(R_t, (x, y))

    circle_t_2d = [apply_stage(p) for p in circle0_2d]
    e1_t_2d = apply_stage(e1_2d)
    e2_t_2d = apply_stage(e2_2d)

    circle_t_3d = [embed_2d_to_3d(p) for p in circle_t_2d]
    e1_t_3d = embed_2d_to_3d(e1_t_2d)
    e2_t_3d = embed_2d_to_3d(e2_t_2d)

    draw_polyline_3d(screen, circle_t_3d, V_COLOR, width=3, closed=True)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_t_3d, V_BASIS_COLOR, width=4)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_t_3d, V_BASIS_COLOR, width=4)


def animate_Sigma_step_2d(circle1_2d, e1_1_2d, e2_1_2d, t):
    """Animate Σ as scaling in ab-plane."""
    s1, s2 = sing_vals
    s1_t = 1.0 + t * (s1 - 1.0)
    s2_t = 1.0 + t * (s2 - 1.0)

    def scale_t(p):
        return (s1_t * p[0], s2_t * p[1])

    circle_t_2d = [scale_t(p) for p in circle1_2d]
    e1_t_2d = scale_t(e1_1_2d)
    e2_t_2d = scale_t(e2_1_2d)

    circle_t_3d = [embed_2d_to_3d(p) for p in circle_t_2d]
    e1_t_3d = embed_2d_to_3d(e1_t_2d)
    e2_t_3d = embed_2d_to_3d(e2_t_2d)

    draw_polyline_3d(screen, circle_t_3d, SIGMA_COLOR, width=3, closed=True)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_t_3d, SIGMA_BASIS_COLOR, width=4)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_t_3d, SIGMA_BASIS_COLOR, width=4)


def animate_U_step_3d(circle2_3d, e1_2_3d, e2_2_3d, t):
    """
    Animate U as a 3D rotation using a quaternion SLERP
    from identity to the quaternion corresponding to U_current.
    """
    # SLERP between identity and precomputed qU_target
    q0 = (1.0, 0.0, 0.0, 0.0)
    q_t = quat_slerp(q0, qU_target, t)
    R_t = quat_to_matrix(q_t)

    def apply_R_t(p):
        return mat3_vec3(R_t, p)

    circle_t = [apply_R_t(p) for p in circle2_3d]
    e1_t = apply_R_t(e1_2_3d)
    e2_t = apply_R_t(e2_2_3d)

    draw_polyline_3d(screen, circle_t, U_COLOR, width=3, closed=True)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_t, U_BASIS_COLOR, width=4)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_t, U_BASIS_COLOR, width=4)
    draw_lift_lines(screen, circle_t, step=16)


# =====================  Calibration (β̂ = V Σ^+ U^T s)  =====================


def compute_theta_hat_and_fit():
    """
    Use current M (3x2), U, Σ, V and s to compute least-squares
    β̂ = (A, B) and s_hat = M β̂.
    """
    global calib_valid, theta_hat, s_hat_vector

    s1, s2 = sing_vals
    if s1 < 1e-10 or s2 < 1e-10:
        calib_valid = False
        theta_hat = None
        s_hat_vector = None
        print("Calibration parameters: (ill-conditioned or undefined)")
        return

    U = U_current
    V = V_current
    s = s_vector

    # Columns of U
    u1 = (U[0][0], U[1][0], U[2][0])
    u2 = (U[0][1], U[1][1], U[2][1])

    # α = U^T s
    alpha1 = vec3_dot(u1, s)
    alpha2 = vec3_dot(u2, s)

    # β' in singular directions
    beta_prime1 = alpha1 / s1
    beta_prime2 = alpha2 / s2

    # β = V β'
    a = V[0][0] * beta_prime1 + V[0][1] * beta_prime2
    b = V[1][0] * beta_prime1 + V[1][1] * beta_prime2
    theta_hat = (a, b)

    # Compute s=s_hat = M β̂
    m11, m12 = M_current[0]
    m21, m22 = M_current[1]
    m31, m32 = M_current[2]
    s_hat_vector = (m11 * a + m12 * b, m21 * a + m22 * b, m31 * a + m32 * b)

    calib_valid = True
    r = vec3_sub(s_vector, s_hat_vector)
    rnorm = vec3_norm(r)
    print(f"Calibration parameters: a = {a:.4g}, b = {b:.4g}, Residual magnitude = {rnorm:.4g}")


# =====================  Main redraw  =====================


def redraw():
    screen.fill(BG)
    draw_grid_and_axes(screen)

    circle0_2d = generate_unit_circle_2d(300)
    e1_2d = (1.0, 0.0)  # a-axis
    e2_2d = (0.0, 1.0)  # b-axis

    circle0_3d = [embed_2d_to_3d(p) for p in circle0_2d]
    draw_polyline_3d(screen, circle0_3d, CIRCLE_COLOR, width=2, closed=True)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e1_2d), BASIS_COLOR, width=3)
    draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e2_2d), BASIS_COLOR, width=3)

    M = M_current
    U = U_current
    V = V_current
    s1, s2 = sing_vals

    # Variables for SVD-step β animation
    svd_anim_star_pos = None
    svd_anim_star_color = None
    svd_in_anim_view = False

    def apply_M(p2):
        x, y = p2
        return (
            M[0][0] * x + M[0][1] * y,
            M[1][0] * x + M[1][1] * y,
            M[2][0] * x + M[2][1] * y,
        )

    if draw_mode == "M_only":
        # Compute M·circle and interpolate between unit circle and M·circle
        circle_M_3d = [apply_M(p) for p in circle0_2d]
        Me1 = apply_M(e1_2d)
        Me2 = apply_M(e2_2d)

        t = a_only_t

        # Interpolated ellipse points
        circle_interp = [
            (
                (1.0 - t) * p0[0] + t * pM[0],
                (1.0 - t) * p0[1] + t * pM[1],
                (1.0 - t) * p0[2] + t * pM[2],
            )
            for p0, pM in zip(circle0_3d, circle_M_3d)
        ]
        draw_polyline_3d(screen, circle_interp, U_COLOR, width=3, closed=True)

        # Interpolated basis vectors from original basis to M·basis
        e1_base_3d = embed_2d_to_3d(e1_2d)
        e2_base_3d = embed_2d_to_3d(e2_2d)

        e1_interp = (
            (1.0 - t) * e1_base_3d[0] + t * Me1[0],
            (1.0 - t) * e1_base_3d[1] + t * Me1[1],
            (1.0 - t) * e1_base_3d[2] + t * Me1[2],
        )
        e2_interp = (
            (1.0 - t) * e2_base_3d[0] + t * Me2[0],
            (1.0 - t) * e2_base_3d[1] + t * Me2[1],
            (1.0 - t) * e2_base_3d[2] + t * Me2[2],
        )

        # Basis of the ellipse is magenta as well
        draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_interp, U_BASIS_COLOR, width=4)
        draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_interp, U_BASIS_COLOR, width=4)

        # Lift lines from interpolated ellipse
        draw_lift_lines(screen, circle_interp, step=16)

        put_status(screen, "Mode: M only — direct interpolation: unit circle → M·circle (UΣVᵀ ellipse) + calibration overlay.")
    else:
        # SVD steps
        Vt = [
            [V[0][0], V[1][0]],
            [V[0][1], V[1][1]],
        ]

        def apply_Vt(p2):
            return mat2_vec2(Vt, p2)

        circle1_2d = [apply_Vt(p) for p in circle0_2d]
        e1_1_2d = apply_Vt(e1_2d)
        e2_1_2d = apply_Vt(e2_2d)

        def apply_sigma(p2):
            return (s1 * p2[0], s2 * p2[1])

        circle2_2d = [apply_sigma(p) for p in circle1_2d]
        e1_2_2d = apply_sigma(e1_1_2d)
        e2_2_2d = apply_sigma(e2_1_2d)

        circle2_3d = [embed_2d_to_3d(p) for p in circle2_2d]
        e1_2_3d = embed_2d_to_3d(e1_2_2d)
        e2_2_3d = embed_2d_to_3d(e2_2_2d)

        def apply_U(p3):
            return mat3_vec3(U, p3)

        circle3_3d = [apply_U(p) for p in circle2_3d]
        e1_3_3d = apply_U(e1_2_3d)
        e2_3_3d = apply_U(e2_2_3d)

        in_anim_view = anim_phase in (1, 2, 3) and (anim_running or anim_frame > 0)
        svd_in_anim_view = in_anim_view

        if in_anim_view:
            t = min(1.0, max(0.0, anim_frame / float(PHASE_FRAMES)))

            # Background partial stages
            if anim_phase > 1:
                draw_polyline_3d(screen, [embed_2d_to_3d(p) for p in circle1_2d], V_COLOR, width=2, closed=True)
                draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e1_1_2d), V_BASIS_COLOR, width=3)
                draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e2_1_2d), V_BASIS_COLOR, width=3)
            if anim_phase > 2:
                draw_polyline_3d(screen, circle2_3d, SIGMA_COLOR, width=2, closed=True)
                draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_2_3d, SIGMA_BASIS_COLOR, width=3)
                draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_2_3d, SIGMA_BASIS_COLOR, width=3)

            # Foreground animated stage
            if anim_phase == 1:
                animate_V_step_2d(circle0_2d, e1_2d, e2_2d, t)
            elif anim_phase == 2:
                animate_Sigma_step_2d(circle1_2d, e1_1_2d, e2_1_2d, t)
            elif anim_phase == 3:
                animate_U_step_3d(circle2_3d, e1_2_3d, e2_2_3d, t)

            # --- β animation in SVD_steps mode ---
            if calib_valid and theta_hat is not None and s_hat_vector is not None:
                a, b = theta_hat
                beta0_2d = (a, b)

                if anim_phase == 1:
                    # Same Vᵀ interpolation as animate_V_step_2d
                    detVt = Vt[0][0] * Vt[1][1] - Vt[0][1] * Vt[1][0]
                    if detVt >= 0:
                        R = Vt
                        F = None
                    else:
                        F = [[1.0, 0.0], [0.0, -1.0]]
                        R = mat2_mul(Vt, F)

                    angle = math.atan2(R[1][0], R[0][0])
                    theta_total = angle
                    theta_t = theta_total * t
                    c = math.cos(theta_t)
                    s = math.sin(theta_t)
                    R_t = [[c, -s], [s, c]]

                    x, y = beta0_2d
                    if F is not None:
                        x, y = x, -y
                    x2 = R_t[0][0] * x + R_t[0][1] * y
                    y2 = R_t[1][0] * x + R_t[1][1] * y
                    svd_anim_star_pos = embed_2d_to_3d((x2, y2))
                    svd_anim_star_color = V_COLOR

                elif anim_phase == 2:
                    # Start from β after full Vᵀ, then interpolate scaling
                    beta1_2d = apply_Vt(beta0_2d)
                    s1_t = 1.0 + t * (s1 - 1.0)
                    s2_t = 1.0 + t * (s2 - 1.0)
                    beta2_t_2d = (s1_t * beta1_2d[0], s2_t * beta1_2d[1])
                    svd_anim_star_pos = embed_2d_to_3d(beta2_t_2d)
                    svd_anim_star_color = SIGMA_COLOR

                elif anim_phase == 3:
                    # β after full Vᵀ and Σ, then partial U via SLERP using qU_target
                    beta1_2d = apply_Vt(beta0_2d)
                    beta2_2d = apply_sigma(beta1_2d)
                    beta2_3d = embed_2d_to_3d(beta2_2d)

                    q0 = (1.0, 0.0, 0.0, 0.0)
                    q_t = quat_slerp(q0, qU_target, t)
                    R_t = quat_to_matrix(q_t)
                    svd_anim_star_pos = mat3_vec3(R_t, beta2_3d)
                    svd_anim_star_color = U_COLOR

            put_status(screen, "Mode: SVD steps — animating Vᵀ / Σ / U (M = UΣVᵀ) + calibration overlay.")
        else:
            # Static: all stages visible
            # Vᵀ
            draw_polyline_3d(screen, [embed_2d_to_3d(p) for p in circle1_2d], V_COLOR, width=2, closed=True)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e1_1_2d), V_BASIS_COLOR, width=3)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), embed_2d_to_3d(e2_1_2d), V_BASIS_COLOR, width=3)
            # ΣVᵀ
            draw_polyline_3d(screen, circle2_3d, SIGMA_COLOR, width=2, closed=True)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_2_3d, SIGMA_BASIS_COLOR, width=3)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_2_3d, SIGMA_BASIS_COLOR, width=3)
            # UΣVᵀ
            draw_polyline_3d(screen, circle3_3d, U_COLOR, width=3, closed=True)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), e1_3_3d, U_BASIS_COLOR, width=4)
            draw_arrow_3d(screen, (0.0, 0.0, 0.0), e2_3_3d, U_BASIS_COLOR, width=4)
            draw_lift_lines(screen, circle3_3d, step=16)

            put_status(screen, "Mode: SVD steps — Vᵀ, ΣVᵀ, UΣVᵀ (M) + calibration overlay.")

    # --- Calibration overlay: s, Mβ, residual, and β motion ---
    # s is always drawn
    draw_star_3d(screen, s_vector, GROUND_TRUTH_COLOR, radius=8, width=2)

    if calib_valid and theta_hat is not None and s_hat_vector is not None:
        a, b = theta_hat
        beta_plane = (a, b, 0.0)

        # Persistent gray β in the ab-plane
        draw_star_3d(screen, beta_plane, BETA_COLOR, radius=7, width=2)

        if draw_mode == "M_only":
            # In M-only mode, animate magenta star from β-plane to Mβ
            t = max(0.0, min(1.0, a_only_t))

            if t <= 0.0:
                # Before animation: show magenta fit point at Mβ and residual from there
                draw_star_3d(screen, s_hat_vector, FIT_POINT_COLOR, radius=7, width=2)
            else:
                # During animation (and at the end): star blends from β-plane to Mβ
                beta_interp = (
                    (1.0 - t) * beta_plane[0] + t * s_hat_vector[0],
                    (1.0 - t) * beta_plane[1] + t * s_hat_vector[1],
                    (1.0 - t) * beta_plane[2] + t * s_hat_vector[2],
                )
                draw_star_3d(screen, beta_interp, FIT_POINT_COLOR, radius=7, width=2)

                # Residual from current animated fit position to s
                residual_vec = vec3_sub(s_vector, beta_interp)
                draw_arrow_3d(screen, beta_interp, residual_vec, RESIDUAL_COLOR, width=2)

        else:
            # SVD_steps mode
            if svd_in_anim_view and svd_anim_star_pos is not None:
                # Animated β following current SVD stage (color matches current ellipse)
                draw_star_3d(screen, svd_anim_star_pos, svd_anim_star_color, radius=7, width=2)
                residual_vec = vec3_sub(s_vector, svd_anim_star_pos)
                draw_arrow_3d(screen, svd_anim_star_pos, residual_vec, RESIDUAL_COLOR, width=2)
            else:
                # Static SVD view: final fit point Mβ in magenta
                draw_star_3d(screen, s_hat_vector, FIT_POINT_COLOR, radius=7, width=2)
                residual_vec = vec3_sub(s_vector, s_hat_vector)
                draw_arrow_3d(screen, s_hat_vector, residual_vec, RESIDUAL_COLOR, width=2)

    put_legend(screen)
    # update_phase_label() # Removed as part of Tkinter cleanup, info is in draw_hud
    pygame.display.flip()


# =====================  Tk callbacks  =====================


def Exit():
    global running
    running = False
    pygame.quit()
    raise SystemExit


def update_svd_params(M):
    global M_current, U_current, V_current, sing_vals, anim_running, anim_phase, anim_frame, a_only_t, qU_target
    U, (s1, s2), V = svd_3x2(M)

    M_current = M
    U_current = U
    V_current = V
    sing_vals = (s1, s2)

    # Precompute quaternion for U rotation
    qU_target = quat_from_rotation_matrix(U_current)

    # Reset animation states
    anim_running = False
    anim_phase = 0
    anim_frame = 0
    a_only_t = 0.0
    compute_theta_hat_and_fit()
    redraw()


def update_ground_truth(s):
    global s_vector
    s_vector = s
    compute_theta_hat_and_fit()
    redraw()


def random_matrix(well_conditioned=True):
    """
    Generate a random 3x2 matrix M via U Σ V^T with random 3D orientation.
    """
    # Random 3D rotation for U
    theta = random.uniform(0.0, 2 * math.pi)
    phi = random.uniform(-0.5 * math.pi, 0.5 * math.pi)
    ax = math.cos(phi) * math.cos(theta)
    ay = math.cos(phi) * math.sin(theta)
    az = math.sin(phi)
    axis = (ax, ay, az)
    angle = random.uniform(0.0, math.pi)
    U_rand = mat3_from_axis_angle(axis, angle)

    # Random 2D rotation for V
    ang_v = random.uniform(0.0, 2 * math.pi)
    cv = math.cos(ang_v)
    sv = math.sin(ang_v)
    V_rand = [[cv, -sv], [sv, cv]]

    # Singular values
    if well_conditioned:
        s1 = random.uniform(0.8, 1.5)
        s2 = random.uniform(0.6, 1.2)
    else:
        s1 = random.uniform(1.0, 2.0)
        s2 = random.uniform(0.01, 0.1)

    Sigma = [[s1, 0.0], [0.0, s2], [0.0, 0.0]]

    Vt = [
        [V_rand[0][0], V_rand[1][0]],
        [V_rand[0][1], V_rand[1][1]],
    ]

    SVt = [
        [
            Sigma[0][0] * Vt[0][0] + Sigma[0][1] * Vt[1][0],
            Sigma[0][0] * Vt[0][1] + Sigma[0][1] * Vt[1][1],
        ],
        [
            Sigma[1][0] * Vt[0][0] + Sigma[1][1] * Vt[1][0],
            Sigma[1][0] * Vt[0][1] + Sigma[1][1] * Vt[1][1],
        ],
        [
            Sigma[2][0] * Vt[0][0] + Sigma[2][1] * Vt[1][0],
            Sigma[2][0] * Vt[0][1] + Sigma[2][1] * Vt[1][1],
        ],
    ]

    M = [
        [U_rand[0][0] * SVt[0][0] + U_rand[0][1] * SVt[1][0] + U_rand[0][2] * SVt[2][0], U_rand[0][0] * SVt[0][1] + U_rand[0][1] * SVt[1][1] + U_rand[0][2] * SVt[2][1]],
        [U_rand[1][0] * SVt[0][0] + U_rand[1][1] * SVt[1][0] + U_rand[1][2] * SVt[2][0], U_rand[1][0] * SVt[0][1] + U_rand[1][1] * SVt[1][1] + U_rand[1][2] * SVt[2][1]],
        [U_rand[2][0] * SVt[0][0] + U_rand[2][1] * SVt[1][0] + U_rand[2][2] * SVt[2][0], U_rand[2][0] * SVt[0][1] + U_rand[2][1] * SVt[1][1] + U_rand[2][2] * SVt[2][1]],
    ]

    update_svd_params(M)


def toggle_draw_mode():
    global a_only_t, anim_running, anim_phase, anim_frame
    global draw_mode

    draw_mode = "SVD_steps" if draw_mode == "M_only" else "M_only"

    anim_running = False
    anim_phase = 0  # Reset SVD animation phase
    anim_frame = 0  # Reset SVD animation frame

    if draw_mode == "M_only":
        a_only_t = 0.0
    redraw()


def toggle_animation():
    global anim_running, anim_phase, anim_frame, a_only_t

    if draw_mode == "M_only":
        # M-only: simple 0 -> 1 interpolation
        if not anim_running:
            # If already finished, restart from beginning
            if a_only_t >= 1.0:
                a_only_t = 0.0
            anim_running = True
        else:
            anim_running = False
        redraw()
        return

    # SVD steps mode
    if draw_mode != "SVD_steps":
        return

    if not anim_running:
        # Start or resume animation from current phase/frame
        if anim_phase == 0:
            anim_phase = 1
            anim_frame = 0
        anim_running = True
    else:
        # Pause, keep current phase and frame
        anim_running = False
    redraw()


def reset_animation():
    global anim_running, anim_phase, anim_frame, a_only_t
    anim_running = False
    anim_phase = 0
    anim_frame = 0
    a_only_t = 0.0  # Reset M-only animation
    redraw()


# =====================  Pygame pump  =====================


def main():
    global running, anim_phase, anim_frame, anim_running, a_only_t, draw_mode

    # Hardcoded initial parameters
    initial_M = [[0.0, 1.0], [1.0, 1.0], [1.5, 1.0]]
    initial_s = (2.0, -0.4, -0.5)
    draw_mode = "SVD_steps"  # or "SVD_steps"

    # Initial setup
    update_svd_params(initial_M)
    update_ground_truth(initial_s)

    clock = pygame.time.Clock()

    print("\n--- Controls ---")
    print("[SPACE] Play/Pause Animation")
    print("[R]     Reset Animation")
    print("[M]     Toggle Mode (M-only / SVD steps)")
    print("[G]     Random Well-Conditioned Matrix")
    print("[B]     Random Ill-Conditioned Matrix")
    print("[S]     Random Ground Truth Vector")
    print("[ESC]   Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    toggle_animation()
                elif event.key == pygame.K_r:
                    reset_animation()
                elif event.key == pygame.K_m:
                    toggle_draw_mode()
                elif event.key == pygame.K_g:
                    random_matrix(well_conditioned=True)
                elif event.key == pygame.K_b:
                    random_matrix(well_conditioned=False)
                elif event.key == pygame.K_s:
                    update_ground_truth((random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)))

        # SVD steps animation
        if anim_running and draw_mode == "SVD_steps":
            anim_frame += 1
            if anim_frame > PHASE_FRAMES:
                anim_frame = 0
                anim_phase += 1
                if anim_phase > 3:
                    anim_phase = 0
                    anim_running = False

        # M-only interpolation animation (plays once from 0->1)
        if anim_running and draw_mode == "M_only":
            a_only_t += 0.02
            if a_only_t >= 1.0:
                a_only_t = 1.0
                anim_running = False

        redraw()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
