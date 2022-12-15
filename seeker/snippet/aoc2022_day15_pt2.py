#date: 2022-12-15T17:09:21Z
#url: https://api.github.com/gists/40b44b179d11350b479748cc0dd0a4f4
#owner: https://api.github.com/users/dossett

import sys
import re
from itertools import pairwise


# Adapted with love from https://www.geeksforgeeks.org/merging-intervals/
def mergeIntervals(intervals) -> list:
    stack = []
    stack.append(intervals[0])
    for i in intervals[1:]:
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
    return stack


max_coord = 4000000
excluded_columns: set[int] = set()
row_dict: dict[int, set[(int, int)]] = {}
for i in range(0, max_coord + 1):
    row_dict[i] = set()

for line in sys.stdin:
    match = re.match(r'^Sensor at x=(-?\d+), y=(-?\d+): closest beacon is at x=(-?\d+), y=(-?\d+)', line)
    sensor_x = int(match.group(1))
    sensor_y = int(match.group(2))
    beacon_x = int(match.group(3))
    beacon_y = int(match.group(4))

    print("Looking at sensor " + str(sensor_x) + " " + str(sensor_y))
    manhattan_distance = abs(sensor_x - beacon_x) + abs(sensor_y - beacon_y)
    # for each row
    for examine_row in range(0, max_coord + 1):
        if examine_row % 100000 == 0:
            print("  " + str(examine_row))
        # Is the row within the manhattan distance of this sensor?
        y_distance = abs(sensor_y - examine_row)
        if y_distance > manhattan_distance:
            pass
        else:
            eliminated_end = sensor_x - (y_distance - manhattan_distance)
            eliminated_start = sensor_x + (y_distance - manhattan_distance)
            row_dict[examine_row].add((eliminated_start, eliminated_end))

for k, row in row_dict.items():
    # Sort the pairs and convert the tuples to a list
    sorted_row = [[p[0], p[1]] for p in sorted(row)]

    # Look for a gap. Gap is when the second part of the first item is less than the first part of the second item
    for pair in pairwise(mergeIntervals(sorted_row)):
        if pair[0][1] + 1 < pair[1][0]:
            beacon_row = k
            beacon_col = pair[0][1] + 1
            freq = beacon_col * 4000000 + beacon_row
            print("Found frequency " + str(freq) + " from " + str(beacon_row) + " " + str(beacon_col))
            exit(0)
