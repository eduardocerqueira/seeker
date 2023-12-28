#date: 2023-12-28T17:03:10Z
#url: https://api.github.com/gists/16bdfe0b8cfa1f0e2aa124553aabd5ae
#owner: https://api.github.com/users/boverby

#!/usr/bin/python3

#  https://www.reddit.com/r/3Dprinting/comments/7ehlfc/python_script_to_find_stl_dimensions/

## updated fot python3 and bookworm

# Python script to find STL dimensions
# Requrements: sudo pip install numpy-stl

## apt install numpy-stl

## initial warning :
## <frozen importlib._bootstrap>:241: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated;
## in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
##  added
##    import warnings
##    warnings.filterwarnings('ignore')


##  pipe delimited file for  sizes and script to find 10 largest
##  find /srv/archives/3Dprinter_backup -name "*.stl" -exec /usr/local/bin/stldim.py {} \; > /tmp/stl_dims.txt
## sample output = /srv/archives/3Dprinter_backup/ArchivedProjects/Zookins_-_Animal_Napkin_Rings/owl.stl|55.998|78.898|10.0|44181.305

## using the data  note: hadron printer has 210x210 heated bed
##  sort -rn -t'|' -k5 /tmp/stl_dims.txt | head -10    # sort by area
##  sort -rn -t'|' -k2 /tmp/stl_dims.txt | head -1    # sort by x    ( was 345)
##  sort -rn -t'|' -k3 /tmp/stl_dims.txt | head -1    # sort by y    ( was 213)
##  sort -rn -t'|' -k4 /tmp/stl_dims.txt | head -1    # sort by z    ( was 158)

import warnings
warnings.filterwarnings('ignore')

import math
import stl
from stl import mesh
import numpy

import os
import sys

if len(sys.argv) < 2:
    sys.exit('Usage: %s [stl file]' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: file %s was not found!' % sys.argv[1])

# this stolen from numpy-stl documentation
# https://pypi.python.org/pypi/numpy-stl

# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz

main_body = mesh.Mesh.from_file(sys.argv[1])

minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)

max_area = (maxx - minx) * (maxy - miny) * (maxz - minz)

# the logic is easy from there

# original output "modified to use str()"
# print ("File:"+ sys.argv[1] )
# print ("X:"+ str(maxx - minx) )
# print ("Y:"+ str(maxy - miny) )
# print ("Z:"+ str(maxz - minz) )

print ( sys.argv[1] + "|" + str(maxx - minx) + "|" + str(maxy - miny) + "|" +  str(maxz - minz)  + "|" +  str(max_area)  )

