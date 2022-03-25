#date: 2022-03-25T17:07:12Z
#url: https://api.github.com/gists/e6755094b0c6191e68b312eb70ef79e6
#owner: https://api.github.com/users/ES-Alexander

from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np

# SETUP
points = np.array([[1,2],[5,9],[-8,-2],[5,-5],[-4,-1]])
xmin, xmax, ymin, ymax = -10, 10, -10, 10
resolution = 300
pixel_size = max(np.abs(np.array([xmax-xmin, ymax-ymin]))) / resolution
radius = 3
# choose a colour map with decent spread, and light+dark ends
cmap = 'nipy_spectral' # 'gist_ncar'

# DATA WRANGLING
x = np.linspace(xmin, xmax, resolution)
y = np.linspace(ymin, ymax, resolution)
X, Y = np.meshgrid(x, y)
img_points = np.c_[X.ravel(), Y.ravel()]
# create a KD Tree of the points of interest (for fast nearest-neighbours)
tree = KDTree(points)
# search for the two nearest neighbours within the radius
dists, indices = tree.query(img_points, k=2,
                            distance_upper_bound=radius+pixel_size)

# PLOTTING
# closest point indices come first
min_inds = indices[:,0]
min_dists = dists[:,0]
# non-matches are set to infinite distance
min_inds[np.isinf(min_dists)] = len(points) # last colour
# set borders to a consistent colour
dist_diffs = np.abs(np.diff(dists, axis=1)).flatten()
inner_borders = dist_diffs < 2 * pixel_size
outer_borders = np.abs(min_dists - radius) < pixel_size
min_inds[inner_borders | outer_borders] = -1 # first colour
# plot the image at the desired location, with some smoothing
plt.imshow(min_inds.reshape(resolution, -1),
           extent=(xmin,xmax,ymin,ymax), origin='lower',
           interpolation='bicubic', interpolation_stage='rgba',
           cmap=cmap)
# plot the points
plt.scatter(points[:,0], points[:,1], 5, 'k')
plt.show()
