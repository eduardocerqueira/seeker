#date: 2025-09-29T17:13:53Z
#url: https://api.github.com/gists/b6c4439eabef61b82fbea53a2646368f
#owner: https://api.github.com/users/AspirantDrago

import csv

FILENAME = 'holes.csv'

with open(FILENAME, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['no', 'kilometer', 'stripe', 'area', 'depth', 'priority'])

    CONTROL_NUMBER = int(input())
    n = int(input())
    for i in range(n):
        kilometer, stripe, area, depth = map(int, input().split())
        priority = int(area * depth > CONTROL_NUMBER)
        writer.writerow([i + 1, kilometer, stripe, area, depth, priority])
