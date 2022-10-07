#date: 2022-10-07T17:20:11Z
#url: https://api.github.com/gists/dae22671e82dcf4bc71a36c1fff3ae4e
#owner: https://api.github.com/users/alisterburt

import numpy as np
from scipy.spatial.transform import Rotation as R

# set up initial orientations of two particles
p0 = np.eye(3)  # oriented same as basis vectors of coord system
p1 = R.from_euler(seq='XYZ', angles=[10, 10, 60], degrees=True)  # slightly rotated out of plane, in plane quite different

# take y vector from p0, project onto y and y from p1
# take y vector because is easier for construction of x-vector with the cross product
p0_y = p0[:, 1]
p1_x = p1[:, 0]
p1_y = p1[:, 1]
p1_z = p1[:, 2]

# make sure p1_y is normalised first, for rotation matrices this is already the case
p0_y_on_p1_x = np.dot(p0_y, p1_y)
p0_y_on_p1_y = np.dot(p0_y, p1_y)


# calculate new y as linear combination of projections and existing basis for xy plane of p1
p1_new_y = p0_y_on_p1_x * p1_x + p0_y_on_p1_y * p1_y
p1_new_x = np.cross(p1_y, p1_z)

p1_final = np.empty((3, 3))
p1_final[:, 0] = p1_new_x
p1_final[:, 1] = p1_new_y
p1_final[:, 2] = p1_z


