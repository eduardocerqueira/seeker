#date: 2023-11-14T16:54:43Z
#url: https://api.github.com/gists/16b1169e71622e3e8c7183a93f8253fa
#owner: https://api.github.com/users/hlb

# Adjusting the code to make cubes larger and add more colors

# Set up the figure and 3D axis again with a clean slate
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

# Increase the size range of the cubes and create a more colorful ink wash effect
def create_cubes_with_color_ink(ax, num_cubes, min_size, max_size, color_map):
    for _ in range(num_cubes):
        size = np.random.rand() * (max_size - min_size) + min_size  # Random size within a range
        # Random position
        x, y, z = np.random.rand(3) * (1 - max_size)
        vertices = gen_cube_vertices(x, y, z, size)
        faces = cube_faces(vertices)

        # Generate random color indices
        color_idx = np.random.rand(len(faces))
        # Create a more vibrant color map
        cmap = plt.get_cmap(color_map)
        face_colors = cmap(color_idx)

        # Add collection of 3D polygons for the cube faces
        ax.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, linewidths=0.5, edgecolors='k', alpha=0.6))

# Create larger and more colorful cubes
create_cubes_with_color_ink(ax, num_cubes=21, min_size=0.05, max_size=0.15, color_map='viridis')

# Set the axes to not be visible, and set the axes limits
ax.set_axis_off()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Show the plot
plt.show()
