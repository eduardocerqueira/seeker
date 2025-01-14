#date: 2025-01-14T17:00:29Z
#url: https://api.github.com/gists/0fcd509cd1d7c6dc2651981510badb99
#owner: https://api.github.com/users/pavel-kirienko

#!/usr/bin/env python3

"""
A simple script for extracting numerical data from plots: load an image and click on it to record point coordinates.
The first click sets the origin, the second sets the unit scale, the subsequent clicks sample the data points.
Close the window to finish.
Usage example:

    ./trace_image.py my_plot.jpg
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

origin = None  # Will hold (x0, y0) in pixel coords
scale_ref = None  # Will hold (dx_pixels, dy_pixels) to define 1 unit in each axis
points = []  # List of collected points in the new coordinate system

fig, ax = plt.subplots()

USAGE = "1st click sets origin; 2nd sets scale; subsequent clicks log points; close when done"

# Load and display the image
img = mpimg.imread(sys.argv[1])
ax.imshow(img)
ax.set_title(USAGE)


def onclick(event):
    """Handle mouse click events."""
    global origin, scale_ref, points
    if event.xdata is None or event.ydata is None:
        return  # Click was outside the image; ignore
    if origin is None:
        origin = (event.xdata, event.ydata)
        print(f"Origin set to: {origin} (pixels)", file=sys.stderr)
    elif scale_ref is None:
        dx_pixels = event.xdata - origin[0]
        dy_pixels = event.ydata - origin[1]
        scale_ref = (dx_pixels, dy_pixels)
        print(f"Scale reference set based on click at: ({event.xdata:.2f}, {event.ydata:.2f})", file=sys.stderr)
        print(f"→ 1 unit in X = {dx_pixels:.4f} pixels,  1 unit in Y = {dy_pixels:.4f} pixels", file=sys.stderr)
    else:
        pixel_x = event.xdata
        pixel_y = event.ydata
        dx_from_origin = pixel_x - origin[0]
        dy_from_origin = pixel_y - origin[1]
        new_x = dx_from_origin / scale_ref[0]
        new_y = dy_from_origin / scale_ref[1]
        points.append((new_x, new_y))
        print(f"{new_x:+012.6f}\t{new_y:+012.6f}")


fig.canvas.mpl_connect("button_press_event", onclick)
plt.tight_layout()
print(USAGE)
plt.show()

out_file = "points.tab"
print("Saving the data into", out_file, file=sys.stderr)
Path(out_file).write_text("\n".join(f"{x:+012.6f}\t{y:+012.6f}" for x, y in points))
