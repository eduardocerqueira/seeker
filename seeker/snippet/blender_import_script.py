#date: 2023-03-13T17:06:21Z
#url: https://api.github.com/gists/23646c8cce64d4cdfe61467da64853c0
#owner: https://api.github.com/users/Gus-The-Forklift-Driver

import math
import bpy


from mathutils import *

D = bpy.data
C = bpy.context
print('=====START====')
with open('./locations.csv', 'r') as file:
    lines = file.readlines()

for line in lines:
    data = line.replace('\n', '').split(',')
    trans = data[0:16]
    name = data[16]
    trans = [float(value) for value in trans]

    trans = Matrix(((trans[0], trans[1], trans[2], trans[3]),
                    (trans[4], trans[5], trans[6], trans[7]),
                    (trans[8], trans[9], trans[10], trans[11]),
                    (trans[12], trans[13], trans[14], trans[15])))
    
    non_transposed = trans
    trans.transpose()

    loc, rot, scale = trans.decompose()

    duplicate = bpy.data.objects[name].copy()
    duplicate.location = (loc.x, -loc.z, loc.y)
    duplicate.rotation_euler = (Matrix.Rotation(math.radians(90), 4, 'X') @ non_transposed).to_euler()
    duplicate.rotation_euler.x -= math.radians(90)

    duplicate.scale = (scale.x, -scale.z, scale.y)
    bpy.context.scene.collection.objects.link(duplicate)
