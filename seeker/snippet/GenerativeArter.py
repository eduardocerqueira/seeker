#date: 2022-09-26T17:18:02Z
#url: https://api.github.com/gists/b4a29b06fe712ff448f2503318f7abca
#owner: https://api.github.com/users/HateFindingNames

import cv2
import numpy as np
import time
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random as rand
import matplotlib.pyplot as plt

# ==== input image here ====
INFILE = "snd2.jpg"
# ==========================
RADII = (5, 10, 25, 30, 35, 40)		# edge sprinkling radii in px
DENSITY = 5							# no. of seeds per edge sprinkling radius
RESIZE = 2							# resizing factor input -> output image
DEBUG = False						# enable/disable windows of in-between-steps
KCOLORS = 16						# color depth of output image
SCREEN_RESOLUTION = [2540, 1440]	# to fit canny threshold finder window to screen

start_time = time.time()

def get_vertices(regions, vertices):
	vert = []
	for region in regions:
		if len(region) > 0 and -1 not in region:
			verts = [tuple(vertices[index]) for index in region]
			vert.append(verts)
	return vert

def voronoi_finite_polygons_2d(vor, radius=None):
	"""Reconstruct infinite Voronoi regions in a
	2D diagram to finite regions.
	Source:
	[https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
	"""
	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")
	new_regions = []
	new_vertices = vor.vertices.tolist()
	center = vor.points.mean(axis=0)
	if radius is None:
		radius = vor.points.ptp().max()
	# Construct a map containing all ridges for a
	# given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))
	# Reconstruct infinite regions
	for p1, region in enumerate(vor.point_region):
		vertices = vor.regions[region]
		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue
		# reconstruct a non-finite region
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]
		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				# finite ridge: already in the region
				continue
			# Compute the missing endpoint of an
			# infinite ridge
			t = vor.points[p2] - vor.points[p1]  # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal
			midpoint = vor.points[[p1, p2]].mean(axis=0)
			direction = np.sign(
				np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius
			new_region.append(len(new_vertices))
			new_vertices.append(far_point.tolist())
		# Sort region counterclockwise.
		vs = np.asarray([new_vertices[v] for v in new_region])
		c = vs.mean(axis=0)
		angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
		new_region = np.array(new_region)[np.argsort(angles)]
		new_regions.append(new_region.tolist())
	return new_regions, np.asarray(new_vertices)

def salty_edges(arr, radius=5, density=2):
	arr = cleaning_borders(arr, radius)
	idx_edges_col, idx_edges_row = np.nonzero(arr)
	edge_pixels = np.column_stack((idx_edges_row, idx_edges_col))
	# edge_pixels = np.array((idx_edges_col, idx_edges_row)).T

	for edge in edge_pixels:
		x, y = edge
		# creating "stamp" matrix filled with zeros and some random 1s
		mask = arr[y - radius:y + radius+1, x - radius:x + radius+1]
		mask[:,:] = 0
		while np.count_nonzero(mask) != density:
			randcol = rand.randrange(0, np.shape(mask)[0], 1)
			randrow = rand.randrange(0, np.shape(mask)[1], 1)

			mask[randcol,randrow] = 1
	return arr

def cleaning_borders(arr, radius):
	# removing edge pixel at the border of inputimage
	arr[0:radius*2+1,:] = 0 # left border
	arr[:,0:radius*2+1] = 0 # upper border
	arr[np.shape(arr)[0] - 1 - radius*2:np.shape(arr)[0],:] = 0 # right border
	arr[:,np.shape(arr)[1] - 1 - radius*2:np.shape(arr)[1]] = 0 # lower border
	return arr

def kmeaning(img, k=10):
	reshaped = img.reshape((-1,3))
	reshaped = np.float32(reshaped)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	_, pixels, colors = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	colors = np.uint8(colors)
	res = colors[pixels.flatten()]
	res = res.reshape((img.shape))
	if DEBUG == True:
		showWindow("Color quantized", res)
	return res, colors

def showWindow(wname, img):
	# taken from https://stackoverflow.com/a/61902334
	# fitting cv2 window to screen
	cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
	if scaling < 1:	# höher als breit
		cv2.resizeWindow(wname, int(0.9*scaling*SCREEN_RESOLUTION[0]), int(0.9*SCREEN_RESOLUTION[1]))
	else:
		cv2.resizeWindow(wname, int(0.9*SCREEN_RESOLUTION[0]), int(0.9*scaling*SCREEN_RESOLUTION[1]))
	cv2.imshow(wname, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def goodCannyThreshs(img, scaling):
	def callback(x):
		pass

	# img = cv2.imread('snd.jpg', 0) #read image as grayscale
	tfinderwname = 'Threshold finder - press ESC to close and use thresholds'
	canny = cv2.Canny(img, 85, 255) 

	# scaling = SCREEN_RESOLUTION[1]/img.shape[1]
	cv2.namedWindow(tfinderwname, cv2.WINDOW_NORMAL) # make a window with name 'image'
	cv2.createTrackbar('Lower', tfinderwname, 0, 255, callback) # lower threshold trackbar
	cv2.createTrackbar('Upper', tfinderwname, 0, 255, callback) # upper threshold trackbar
	# cv2.createTrackbar('Algo', tfinderwname, False, True, callback)
	cv2.createTrackbar('Aperture', tfinderwname, 0, 2, callback)

	while(1):
		numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
		if numpy_horizontal_concat.shape[1] > SCREEN_RESOLUTION[0]:
			dblscaling = SCREEN_RESOLUTION[0]/numpy_horizontal_concat.shape[1]
		elif numpy_horizontal_concat.shape[0] > SCREEN_RESOLUTION[1]:
			dblscaling = SCREEN_RESOLUTION[1]/numpy_horizontal_concat.shape[0]
		else:
			dblscaling = SCREEN_RESOLUTION[1]/numpy_horizontal_concat.shape[0]
		new_width = dblscaling*numpy_horizontal_concat.shape[1]
		new_height = dblscaling*numpy_horizontal_concat.shape[0]
		cv2.resizeWindow(tfinderwname, int(new_width-(0.5*SCREEN_RESOLUTION[0])), int(new_height-(0.5*SCREEN_RESOLUTION[1])))
		cv2.imshow(tfinderwname, numpy_horizontal_concat)
		k = cv2.waitKey(1) & 0xFF
		if k == 27: # escape key
			break
		l = cv2.getTrackbarPos('Lower', tfinderwname)
		u = cv2.getTrackbarPos('Upper', tfinderwname)
		# a = cv2.getTrackbarPos('Algo', tfinderwname)
		apsize = cv2.getTrackbarPos('Aperture', tfinderwname)
		a = False
		if apsize == 0:
			apsize = 3
		elif apsize == 1:
			apsize = 5
		elif apsize == 2:
			apsize = 7
		# if a == 0:
		# 	canny = cv2.Canny(img, l, u)
		# else:
		# 	canny = cv2.Canny(img, l, u, L2gradient=True)
		canny = cv2.Canny(img, l, u, apertureSize=apsize, L2gradient=a)
	print('Thresholds:\nLower {}\nUpper {}'.format(l, u))

	cv2.destroyAllWindows()
	return l, u

img = cv2.imread(INFILE)
img = cv2.resize(img, (int(img.shape[1]*RESIZE),int(img.shape[0]*RESIZE)))
scaling = SCREEN_RESOLUTION[0]/img.shape[0]

if DEBUG == True:
	showWindow('Original', img)
# kimg, colors = kmeaning(img, KCOLORS)
# h, w, _ = np.shape(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if DEBUG == True:
	showWindow("Grayscale", img_gray)
print("Image opened.")

l, u = goodCannyThreshs(img_gray, scaling)
img_edges = cv2.Canny(img_gray, l, u, L2gradient=True) # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# if DEBUG == True:
# 	showWindow("Edges", img_edges)
# print("Edges found.")
kimg, colors = kmeaning(img, KCOLORS)
h, w, _ = np.shape(img)

# # Generating some overall noise
# noise = np.random.randint(0, 60000, size=(h,w))
# noise = np.where(noise < 2, 255, 0)
# im_noise = Image.fromarray(noise, "L")
# # im_noise.show()
# # idx_noise_col, idx_noise_row = np.nonzero(noise)
# noisepx = np.column_stack((np.nonzero(noise)[1], np.nonzero(noise)[0]))
# print("General noise generated.")

# Sprinkle some salt along the edges found above
for i, radius in enumerate(RADII):
	print("Iteration ", i)
	img_edges = salty_edges(img_edges, radius, DENSITY)
	img_edges = img_edges * 255
	if DEBUG == True:
		showWindow("Seed distribution for radius ".format(radius), img_edges)
	print("Edges salted with radius {}".format(radius))

	edgepx_y, edgepx_x = np.nonzero(img_edges)
	edgepx = np.column_stack((edgepx_x, edgepx_y))

	'''
	Getting clustered color from generated seeds
	'''
	colored_seeds = {tuple(seed): kimg[seed[1],seed[0]] for seed in edgepx}

	if np.shape(edgepx)[0] >= 4:
		vor = Voronoi(edgepx)
		regions, vertices = voronoi_finite_polygons_2d(vor)
		polygons = get_vertices(regions, vertices)

		vor_im = Image.new("RGB", (w,h))
		vor_im_draw = ImageDraw.Draw(vor_im)
		for j, polygon in enumerate(polygons):
			if len(polygon) > 2:
				color = colored_seeds[tuple(edgepx[j])]
				color = tuple((color[2], color[1], color[0]))
				vor_im_draw.polygon(polygon, fill=color, outline='black', width=1)
		# for px in edgepx:
		# 	vor_im_draw.point(px, fill="yellow")
		print("Segmentation done with radius {}".format(radius))
		vor_im.save("outfiles\\{}_vor_radius{}_l{}_u{}.png".format(INFILE, radius, l, u))
		# vor_im.show()
	else:
		print("Not enough points left. No diagram generated.")

print("Done! Time taken {}".format(time.time() - start_time))