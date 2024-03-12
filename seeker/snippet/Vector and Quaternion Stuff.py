#date: 2024-03-12T17:00:25Z
#url: https://api.github.com/gists/9d2796279ed48fd1ec7b78e7bcedc89e
#owner: https://api.github.com/users/s4lt3d

# Helps visualize tough projection problems. 

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:50:51 2024

@author: walter
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euler_to_quaternion(roll, pitch, yaw):
    roll = np.radians(roll)  
    pitch = np.radians(pitch) 
    yaw = np.radians(yaw)  
    """
    Convert Euler angles to a quaternion.

    Parameters:
    - roll: Rotation around the x-axis in radians.
    - pitch: Rotation around the y-axis in radians.
    - yaw: Rotation around the z-axis in radians.

    Returns:
    - A numpy array representing the quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def quaternion_from_angle_axis(angle, axis ):
    """Create a quaternion from an axis and rotation angle."""
    angle = np.radians(angle)
    axis = normalize(axis)
    s = np.sin(angle / 2)
    w = np.cos(angle / 2)
    x, y, z = axis * s
    return np.array([w, x, y, z])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def rotate_quaternion(q, axis, angle):
    """Rotate a quaternion around an axis by a given angle."""
    q_rot = quaternion_from_angle_axis(angle, axis)
    return quaternion_multiply(q_rot, q)

def quaternion_conjugate(q):
    """Calculate the conjugate of a quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(v, q):
    """Rotate a vector by a quaternion."""
    v_quat = np.array([0] + v.tolist())  # Convert vector to quaternion form
    q_conj = quaternion_conjugate(q)
    v_rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return v_rotated_quat[1:]  # Return only the vector part

def quaternion_to_euler(q):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)

    Parameters:
    - q: A numpy array representing the quaternion [w, x, y, z].

    Returns:
    - A tuple containing the Euler angles (roll, pitch, yaw) in radians.
    """
    # Extract the quaternion components
    w, x, y, z = q

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.degrees([roll_x, pitch_y, yaw_z])

# Example: Convert quaternion back to Euler angles
q = np.array([0.7071068, 0, 0.7071068, 0])  # Example quaternion

q = euler_to_quaternion(30, 40, 50)
roll, pitch, yaw = quaternion_to_euler(q)

  # 


def plot_vectors_3d(vectors, offsets, colors=None):
    """
    Plots vectors with specified offsets and colors in 3D space, with Y axis as up.
    
    Parameters:
    - vectors: A list of tuples/lists, where each tuple/list represents a vector in the form (x, y, z).
    - offsets: A list of tuples/lists, where each tuple/list represents the starting point of the vector.
    - colors: A list of colors for each vector. If None, a default color will be used for all.
    """
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    
    # If no specific colors are provided, use a default 'r' for all vectors
    if colors is None:
        colors = ['r']*len(vectors)
    
    for (vector, offset, color) in zip(vectors, offsets, colors):
        # Swap Y and Z for plotting to make Y the up-direction
        ax.quiver(offset[0], offset[2], offset[1], vector[0], vector[2], vector[1], color=color)
    
    ax.set_xlim([-5, 5])
    ax.set_zlim([-5, 5])  # This now represents the Y limits
    ax.set_ylim([-5, 5])  # This now represents the Z limits
    ax.set_xlabel('X axis')
    ax.set_zlabel('Y axis')  # Swapped labels
    ax.set_ylabel('Z axis')  # Swapped labels
    plt.title('3D Vector Plot with Y Up')
    plt.show()

# Example vectors
v = (1, 0, 0)
w = (0, 1, 0)

#calculate_and_plot(v, w)

up = np.array((0,1,0))

object_euler = np.array((0,0,0))

object_rot = euler_to_quaternion(object_euler[0], object_euler[1], object_euler[2])
object_vec = np.array((0,0,1))
object_pos = np.array((0,0,0))

rot_q = quaternion_from_angle_axis(90, up)

object_vec = rotate_vector_by_quaternion(object_vec, rot_q)

plot_vectors_3d([object_vec], [object_pos], ['r'])





player_pos = np.array((2,0,-1))
object_pos = np.array((0,0,0))

player_vec = np.array((1,0,1))
object_vec = np.array((0,0,1))

player_vec = normalize(player_vec)

new_vec = object_vec * (np.dot(player_vec, object_vec) / np.dot(object_vec, object_vec))
new_vec = new_vec / np.linalg.norm(new_vec)

vectors_3d = [player_vec, object_vec, new_vec]
offsets_3d = [player_pos, object_pos, player_pos]
colors = ['r', 'b', 'g']  # Optional: specify colors for each vector
plot_vectors_3d(vectors_3d, offsets_3d, colors)




