#date: 2024-07-01T16:35:46Z
#url: https://api.github.com/gists/e51f1b63b44c2d3a4b250bff7599aa92
#owner: https://api.github.com/users/caguero

import matplotlib.pyplot as plt

# Data for selected series
x_values = list(range(1, 9))
y_values_external_gz_ros = [1212, 1156, 1239, 1349, 1236, 1201, 1149, 1197]
y_values_gz_bridge_external_ros = [1079, 1069, 1022, 1030, 1044, 1035, 1024, 1022]
y_values_gz_bridge_ros = [593, 539, 535, 538, 536, 550, 593, 612]

# Plot
plt.figure(figsize=(12, 8))

# Adding each dataset to the plot
plt.plot(x_values, y_values_external_gz_ros, marker='d', label='External Gz Transport, external bridge, external ROS node', linestyle='-')
plt.plot(x_values, y_values_gz_bridge_external_ros, marker='*', label='(Gz Transport + bridge), external ROS node', linestyle='--')
plt.plot(x_values, y_values_gz_bridge_ros, marker='P', label='(Gz Transport + bridge + ROS node)', linestyle='-.')

# Labels and title
plt.xlabel('Data Point Index')
plt.ylabel('Time (ms.)')
plt.title('Gazebo to ROS performance - 1000 images received (all examples running within a ROS node)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()