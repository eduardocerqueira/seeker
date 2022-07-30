#date: 2022-07-30T19:20:31Z
#url: https://api.github.com/gists/50d2d835f46fe9a456e7bafb0e24a310
#owner: https://api.github.com/users/Chim-SO

# Create numpy pointcloud:
number_points = 2000
pcd_np = np.random.rand(number_points, 3)

# Convert to Open3D.PointCLoud:
pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)  # set pcd_np as the point cloud points

# Visualize:
o3d.visualization.draw_geometries([pcd_o3d])