#date: 2025-03-18T17:08:11Z
#url: https://api.github.com/gists/e810448628d6767e4c85cfe228bc9b63
#owner: https://api.github.com/users/trbarron

import matplotlib.pyplot as plt
import numpy as np

def generate_semicircle(center_x, center_y, radius, num_points=1000):
    # Generate points for the curved part (top half of the circle)
    theta = np.linspace(0, np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    
    # Add the flat bottom to close the shape
    x = np.append(x, [center_x - radius, center_x - radius])
    y = np.append(y, [center_y, center_y])
    
    return x, y

def plot_inside_semicircle(ax, center_x, center_y, radius, function_to_plot, num_points=1000):
    # Create a grid of points
    x = np.linspace(center_x - radius, center_x + radius, num_points)
    y = np.linspace(center_y, center_y + radius, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate squared distance from center
    dist_squared = (X - center_x)**2 + (Y - center_y)**2
    
    # Create a mask for points inside the semicircle
    mask = (dist_squared <= radius**2) & (Y >= center_y)
    
    # Apply the provided function to get values for coloring
    Z = np.zeros_like(X)
    Z[mask] = function_to_plot(X[mask], Y[mask])
    
    # Make points outside the semicircle NaN to avoid showing contours there
    Z_masked = Z.copy()
    Z_masked[~mask] = np.nan
    
    # Plot the result using a diverging colormap
    im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='twilight_shifted_r', vmin=0, vmax=1)
    
    ax.contour(X, Y, Z_masked, levels=[0.5], colors='black', linewidths=1)
    
    return im, Z_masked

# Function that compares y-value (distance from bottom) to distance to edge
def distance_comparison(x, y):
    dist_from_bottom = y
    
    # Distance to the curved edge is (radius - distance from center)
    radius = 1.0
    dist_from_center = np.sqrt(x**2 + y**2)
    dist_to_curved_edge = radius - dist_from_center
    
    epsilon = 1e-10
    
    # Direct ratio of distance from bottom to sum of distances
    # This ensures the ratio is 0.5 when both distances are equal
    ratio = dist_from_bottom / (dist_from_bottom + dist_to_curved_edge + epsilon)
    
    return ratio

# Main visualization
def create_distance_ratio_plot():
    # Set blue background style for the figure
    plt.style.use('dark_background')
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#1a3a63')
    
    # Set the semicircle parameters
    center_x, center_y = 0, 0
    radius = 1

    # Plot the semicircle boundary
    semicircle_x, semicircle_y = generate_semicircle(center_x, center_y, radius)
    ax.plot(semicircle_x, semicircle_y, 'white', linewidth=2)

    # Plot inside the semicircle with our distance comparison function and add 0.5 contour
    im, Z = plot_inside_semicircle(ax, center_x, center_y, radius, distance_comparison)
    
    # Create colorbar with size matching the plot height
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, label='Distance Ratio')
    
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    # Set equal aspect ratio and labels with white text
    ax.set_aspect('equal')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_title('Distance Ratio to Beach with contour', color='white', fontsize=14)
    
    # Set tick colors to white
    ax.tick_params(colors='white')

    plt.tight_layout()
    
    # Update title to match your screenshot
    ax.set_title('Distance Ratio to the Beach with 0.5 Contour', color='white', fontsize=14)
    return fig

fig1 = create_distance_ratio_plot()
plt.figure(fig1.number)
plt.show()