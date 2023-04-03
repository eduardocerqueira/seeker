#date: 2023-04-03T16:48:37Z
#url: https://api.github.com/gists/d352b8c17566241c5ca366e23e498f32
#owner: https://api.github.com/users/yulleyi

import numpy as np
import matplotlib.pyplot as plt

# Parameters for Rose curve
A = 1
k = 5

# Generate theta values from 0 to 2*k*pi with 1000 points
theta = np.linspace(0, 2 * k * np.pi, 1000)

# Calculate r values using the Rose curve equation
r = A * np.cos(k * theta)

# Create the polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(1, 1, 1, polar=True)

# Plot the Rose curve
ax.plot(theta, r, linewidth=2)

# Set plot properties
ax.set_title("Rose Curve Flower")
ax.set_rticks([])  # Disable radial ticks
ax.set_xticks(np.pi * np.arange(0, 2, 1 / k))  # Set angular ticks based on k
ax.set_xticklabels([])  # Disable angular tick labels
ax.grid(True)

# Show the plot
plt.show()