#date: 2022-11-21T13:50:29Z
#url: https://api.github.com/gists/cdafc55c2f148a4533b1b8f990b82f8e
#owner: https://api.github.com/users/fredley

import gpxpy
from haversine import haversine
from pyproj import Transformer
import sys
from svgpathtools import Path, Line
import svgwrite

fnames = sys.argv[1:]

if not fnames:
    print("No files specified")
    sys.exit(1)

TRANSFORM = Transformer.from_crs("EPSG:4326", "EPSG:3857")
STROKE = svgwrite.rgb(0, 0, 0)

BOUNDS_LONG_LEFT = -0.4982039
BOUNDS_LONG_RIGHT = 0.0857954
BOUNDS_LAT_TOP = 51.6511052
BOUNDS_LAT_BOTTOM = 51.3706793

# Discontinuity detection, to remove GPS cutouts
MAX_DISTANCE_BETWEEN_POINTS_M = 100

inf = float('inf')

top = -1 * inf
bottom = inf
left = inf
right = -1 * inf

paths = []

def in_bounds(p):
    # print(f"{BOUNDS_LAT_BOTTOM} < {p.latitude} < {BOUNDS_LAT_TOP}")
    # print(f"{BOUNDS_LONG_LEFT} < {p.longitude} < {BOUNDS_LONG_RIGHT}")
    return (
        BOUNDS_LAT_BOTTOM < p.latitude < BOUNDS_LAT_TOP
        and BOUNDS_LONG_LEFT < p.longitude < BOUNDS_LONG_RIGHT
    )

def distance(p1, p2):
    distance_km = haversine((p1.latitude, p1.longitude), (p2.latitude, p2.longitude))
    return distance_km / 1000

for f in fnames:
    with open(f) as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    for track in gpx.tracks:
        for segment in track.segments:
            path = []
            prev_point = None
            for point in segment.points:
                if not in_bounds(point):
                    # print("Skipping")
                    continue
                p = TRANSFORM.transform(point.longitude, point.latitude)
                bottom = min(p[1], bottom)
                top = max(p[1], top)
                left = min(p[0], left)
                right = max(p[0], right)
                if prev_point is not None:
                    # Check speed
                    if distance(prev_point, point) > MAX_DISTANCE_BETWEEN_POINTS_M:
                        print(f"Discontinuity in {f}")
                        if len(path) > 1:
                            paths.append(path)
                        path = []
                prev_point = point
                path.append(p)
            if len(path) > 1:
                paths.append(path)

if not paths:
    print(f"No paths in bounds!")
    sys.exit(1)

print(f"Paths: {len(paths)}")

width = abs(right - left)
height = abs(top - bottom)

aspect_ratio = width / height

print(f"Height: {height}, Width: {width}, Ratio: {aspect_ratio}")

max_y = 1000
max_x = max_y * aspect_ratio

print(max_x, max_y)

scale_x = lambda x: round(max_x - 1 * (x - left) * max_x / (right - left), 2) * 1j
scale_y = lambda y: round((y - bottom) * max_y / (top - bottom), 2)

svg = svgwrite.Drawing('output.svg', size=(max_y, max_x), profile='full')

for path in paths:
    lines = []
    current_point = None
    for p in path:
        point = scale_x(p[0]) + scale_y(p[1])
        if current_point is not None:
            lines.append(Line(current_point, point))
        current_point = point
    svg.add(svg.path(d=Path(*lines).d(), fill=svgwrite.rgb(255, 255, 255), stroke=STROKE, fill_opacity=0))

svg.save()

print("Done!")
