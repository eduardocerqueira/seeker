#date: 2022-01-11T16:54:54Z
#url: https://api.github.com/gists/2b94312b2ff85423d4344c84a3aa9535
#owner: https://api.github.com/users/delta2golf

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Stephan Hügel on 2017-03-02
The MIT License (MIT)
Copyright (c) 2017 Stephan Hügel
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# requires scipy, scikit-image, shapely, geojson and dependencies
# this script is simple, but "heavy" on packages bc scipy and scikit

from imageio import imread
from skimage import measure
from skimage.color.colorconv import rgb2gray, rgba2rgb
from shapely.geometry import shape, Point, Polygon, LineString
import geojson

# read a PNG
polypic = imread("poly.png")
# convert to greyscale if need be
gray = rgb2gray(rgba2rgb(polypic))

# find contours
# Not sure why 1.0 works as a level -- maybe experiment with lower values
contours = measure.find_contours(gray, 1.0)

# build polygon, and simplify its vertices if need be
# this assumes a single, contiguous shape
# if you have e.g. multiple shapes, build a MultiPolygon with a list comp

# RESULTING POLYGONS ARE NOT GUARANTEED TO BE SIMPLE OR VALID
# check this yourself using e.g. poly.is_valid
poly = Polygon(contours[0]).simplify(1.0)

# write out to cwd as JSON
with open("polygon.json", "w") as f:
    f.write(geojson.dumps(poly))
