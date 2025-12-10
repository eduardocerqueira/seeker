#date: 2025-12-10T16:49:02Z
#url: https://api.github.com/gists/c2b8229ce6a24133068d53077fe99e1b
#owner: https://api.github.com/users/AspirantDrago

import math

RADIUS = 100


def polygon(n):
    result = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = math.cos(angle) * RADIUS
        y = math.sin(angle) * RADIUS
        result.append((
            int(round(x)),
            int(round(y))
        ))
    return result


def calc_angles(points, n):
    angles = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        x1 = p2[0] - p1[0]
        x2 = p3[0] - p2[0]
        y1 = p2[1] - p1[1]
        y2 = p3[1] - p2[1]
        v1 = math.sqrt(x1 ** 2 + y1 ** 2)
        v2 = math.sqrt(x2 ** 2 + y2 ** 2)
        angle = math.acos((x1 * x2 + y1 * y2) / (v1 * v2))
        angles.append(180 - angle * 180 / math.pi)
    return angles


def start1():
    n = int(input())
    for x, y in polygon(n):
        print(x, y)


def start2():
    n2 = int(input())
    points2 = [
        list(map(float, input().split()))
        for _ in range(n2)
    ]
    angles = calc_angles(points2, n2)
    angles.sort()
    del angles[:2]
    sr_angle = sum(angles) / len(angles)
    n = - 360 / (sr_angle - 180)
    print(int(round(n)))


def main():
    s = input()
    t = int(input())
    for _ in range(t):
        if s == 'draw':
            start1()
        else:
            start2()


main()
