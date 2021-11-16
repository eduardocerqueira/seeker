#date: 2021-11-16T17:06:13Z
#url: https://api.github.com/gists/6d90a9a96e0c915dd4194789c24e305f
#owner: https://api.github.com/users/ad-1

import numpy as np
import matplotlib.pyplot as plt
from vector_3d import Vector3D
from e123 import r1, r2, r3, e123_dcm

# ============================================================
# configure plot

plt.style.use('default')  # dark_background
plt.rc('axes', edgecolor='grey')
plt.rcParams.update({"axes.grid": True, "grid.color": "grey"})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, 0], [0, 0], [0, 0], 'ko', label='Origin')
ax.plot([-1, 1], [0, 0], [0, 0], 'k-', lw=0)
ax.plot([0, 0], [-1, 1], [0, 0], 'k-', lw=0)
ax.plot([0, 0], [0, 0], [-1, 1], 'k-', lw=0)

# ax.set_axis_off()
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([-1, 0, 1])


# ============================================================
# helper functions


# convert degrees to radians
def d2r(degrees):
    return degrees * (np.pi / 180)


# create list of vectors with linearly spaces components
def linspace_vectors(vec1: Vector3D, vec2: Vector3D, step=90):
    return [Vector3D(c1, c2, c3, origin=vec2.origin, color=vec2.color, text=vec2.text) for
            (c1, c2, c3) in zip(np.linspace(vec1.x, vec2.x, step), np.linspace(vec1.y, vec2.y, step), np.linspace(vec1.z, vec2.z, step))]


# plot array of vectors
def plot_vectors(vector_set):
    for vs in vector_set:
        ax.add_artist(vs.arrow)
        ax.text(vs.x, vs.y, vs.z, vs.text, size=12, color=vs.color)
        plt.pause(0.05)


# animate vectors
def animate_vectors(vector_set):
    for item in vector_set:
        artists = []
        for vs in item:
            vector_artist = ax.add_artist(vs.arrow)
            text_artist = ax.text(vs.x, vs.y, vs.z, vs.text, size=12, color=vs.color)
            artists.extend((vector_artist, text_artist))
        plt.pause(0.05)
        for artist in artists:
            artist.remove()


# get axis unit vector from dcm rows
def get_vector_from_dcm(dcm, axis, o, label, color):
    """
    dcm: direction cosine matrix (3x3 numpy array)
    axis: basis vector to extract (0 = x, 1 = y, 2 = z)
    """
    return Vector3D(dcm[axis, :][0], dcm[axis, :][1], dcm[axis, :][2], origin=o, text=label, color=color)


# ============================================================
# simulation harness parameters

# inertial frame origin
origin = Vector3D(0, 0, 0)

initial_frame_color = 'k'
# inertial frame X, Y, Z axes
x = Vector3D(1, 0, 0, origin, 'X', color=initial_frame_color)
y = Vector3D(0, 1, 0, origin, 'Y', color=initial_frame_color)
z = Vector3D(0, 0, 1, origin, 'Z', color=initial_frame_color)

# plot initial axes
plot_vectors([x, y, z])

# initialize axes update variables
x_prime, y_prime, z_prime = None, None, None

# initialize DCM variables
q, q_prev = None, None

# ============================================================
# define euler transformation

# Euler 123 Sequence
transformations = [r1, r2, r3]

# euler angles
phi = 30  # roation angle about axis 1 (phi)
theta = 45  # rotation angle about axis 2 (theta)
psi = 55  # rotation angle about axis 3 (psi)

euler_angles = (phi, theta, psi)

frame_colors = ['orange', 'green', 'blue']

# ============================================================
# visualize frame transformation

# propogate system
for i, transformation in enumerate(transformations):

    # array of vectors to animate
    intermediate_vectors = []

    # get rotation angle
    rotation_angle = euler_angles[i]

    # iterate over rotation angle
    for angle in np.linspace(0, rotation_angle, np.abs(rotation_angle)):

        # calculate DCM at each step from 0 to rotation angle
        q = transformation(d2r(angle))

        # chain DCM transformation matrices
        if q_prev is not None:
            q = np.dot(q, q_prev)

        # extract updates x, y and z axes coordinates from DCM
        x_prime = get_vector_from_dcm(q, 0, origin, f'x\'', frame_colors[i])
        y_prime = get_vector_from_dcm(q, 1, origin, f'y\'', frame_colors[i])
        z_prime = get_vector_from_dcm(q, 2, origin, f'z\'', frame_colors[i])

        intermediate_vectors.append((x_prime, y_prime, z_prime))

    # store final DCM from previous transformation
    q_prev = q

    animate_vectors(intermediate_vectors)
    # plot_vectors((x_prime, y_prime, z_prime))

# plot final axes
plot_vectors((x_prime, y_prime, z_prime))

# ============================================================
# DCM validation through direct computation

q_validation = e123_dcm(psi=d2r(psi), theta=d2r(theta), phi=d2r(phi))
plot_vectors((get_vector_from_dcm(q_validation, 0, origin, 'x\'', 'purple'),
              get_vector_from_dcm(q_validation, 1, origin, 'y\'', 'purple'),
              get_vector_from_dcm(q_validation, 2, origin, 'z\'', 'purple')))

assert q.all() == q_validation.all()

# ============================================================
#  show plot
plt.show()


# ***************************
# NEW FILE e123.py
# ***************************


import numpy as np

# ============================================================
# principal rotations


# rotation of phi is about the 1st axis (=X-axis) of the inertial frame
def r1(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), np.sin(phi)],
                     [0, -np.sin(phi), np.cos(phi)]])


# second rotation of theta is about the 2nd axis (=Y-axis) of the first intermediate frame
def r2(theta):
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]])


# third rotation of psi is about the 3rd axis (=Z-axis) of the second intermediate frame
def r3(psi):
    return np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])


# ====================================================
# e123 rotation sequence

def q11(psi, theta):
    return np.cos(psi) * np.cos(theta)


def q12(psi, theta, phi):
    return np.cos(psi) * np.sin(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)


def q13(psi, theta, phi):
    return -np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)


def q21(psi, theta):
    return - np.sin(psi) * np.cos(theta)


def q22(psi, theta, phi):
    return -np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)


def q23(psi, theta, phi):
    return np.sin(psi) * np.sin(theta) * np.cos(phi) + np.cos(psi) * np.sin(phi)


def q31(theta):
    return np.sin(theta)


def q32(theta, phi):
    return - np.cos(theta) * np.sin(phi)


def q33(theta, phi):
    return np.cos(theta) * np.cos(phi)


def e123_dcm(psi, theta, phi):
    return np.array([[q11(psi, theta), q12(psi, theta, phi), q13(psi, theta, phi)],
                     [q21(psi, theta), q22(psi, theta, phi), q23(psi, theta, phi)],
                     [q31(theta), q32(theta, phi), q33(theta, phi)]])
