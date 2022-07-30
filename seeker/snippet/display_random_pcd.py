#date: 2022-07-30T19:13:38Z
#url: https://api.github.com/gists/d206ce08bdec352c3bdbf5c294c2070d
#owner: https://api.github.com/users/Chim-SO

# Create Figure:
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2])
# label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Random Point Cloud")
# display:
plt.show()
