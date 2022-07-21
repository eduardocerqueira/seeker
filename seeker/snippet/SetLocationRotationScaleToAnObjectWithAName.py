#date: 2022-07-21T17:17:36Z
#url: https://api.github.com/gists/9f5425a62c92e0299188ea5343b7dc9a
#owner: https://api.github.com/users/gladiopeace

import bpy

def set_location(name, xyz):
     bpy.data.objects[name].location = xyz

def set_euler_rotation(name, xyz):
    bpy.data.objects[name].rotation_euler = xyz

def set_scale(name, xyz):
    bpy.data.objects[name].scale = xyz

set_location("PEPE", (0, 0, 5))
set_euler_rotation("PEPE", (3.1415 * 0.5, 3.1415 * 0.5 * 3, 3.1415 * 0.5 * 0.5))
set_scale("PEPE", (1,2,3))