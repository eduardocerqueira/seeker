#date: 2023-12-20T17:09:00Z
#url: https://api.github.com/gists/06e58a1602e6fbda4b03ae0023d6229d
#owner: https://api.github.com/users/TimSmith714

from tree import RGBXmasTree
from time import sleep

tree = RGBXmasTree()

bottomRow = [0,16,15,6,12,24,19,7]
middleRow = [1,17,14,5,11,23,20,8]
topRow = [2,18,13,4,10,22,21,9]
star = [3]
rows = [bottomRow, middleRow, topRow, star]
colorStep = 0.0039

def setRow(lights, r, g, b):
    for pixel in lights:
        tree[pixel].color = (r,g,b)

tree.color = (1,0,0)
tree.brightness = 0.1

while 1==1:

    for row in (rows):
        for r in range (1,255,15):
            #print("color:", 1 - colorStep * r, colorStep * r)
            setRow(row, 1 - colorStep * r, colorStep * r, 0)

    sleep(4)

    for row in reversed(rows):
        for r in range (1,255,15):
            setRow(row, colorStep * r, 1 - colorStep * r, 0)

    sleep(4)
