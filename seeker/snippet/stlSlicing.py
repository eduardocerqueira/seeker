#date: 2021-09-15T16:52:33Z
#url: https://api.github.com/gists/9b0507d32e3d81d5e7f7c34ac221207c
#owner: https://api.github.com/users/iamargentum

import trimesh
import numpy as np

myMesh = trimesh.load_mesh('assignment.stl')

basePlane = input('please input base plane\n1. bottom\n2. left\n3. right\n4.front\n5. right\n6. top\n')

if (basePlane == '1'):
    planeNormal = [0, 0, 1]
if (basePlane == '6'):
    planeNormal = [0, 0, -1]
if (basePlane == '2'):
    planeNormal = [0, 1, 0]
if (basePlane == '3'):
    planeNormal = [0, -1, 0]
if (basePlane == '4'):
    planeNormal = [1, 0, 0]
if (basePlane == '5'):
    planeNormal = [-1, 0, 0]
else:
    print('selecting default base plane')
    planeNormal = [0, 0, 1]

planeOrigin = [0, 0, 0]

height = int(input('please input height - '))
for i in range(len(planeNormal)):
    planeOrigin[i] = planeNormal[i]*height

try:
    slicedMesh = myMesh.section(plane_origin=planeOrigin, plane_normal=planeNormal)
    slicedMesh.show()
except AttributeError:
    print('geometry does not exist in the slicing plane')