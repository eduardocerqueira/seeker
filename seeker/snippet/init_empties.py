#date: 2025-08-04T16:49:49Z
#url: https://api.github.com/gists/563fd9afee7a3bbf370c269dd9d63ca5
#owner: https://api.github.com/users/Birdacious

import bpy

arm_data = bpy.data.armatures.new("FingerArmature")
arm_obj = bpy.data.objects.new("FingerArmatureObj", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode='EDIT')

def create_empty(name):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'SPHERE'
    empty.empty_display_size = 0.05
    empty.location = (0, 0, 0)
    bpy.context.collection.objects.link(empty)
    return empty

empties = {}
for h in range(0, 2):  # hand
    for d in range(0, 5):  # digit
        for b in range(0, 4):  # joint
            name_p = f"{h}{d}{b}p"
            name_n = f"{h}{d}{b}n"
            ep = create_empty(name_p)
            en = create_empty(name_n)
            ep.location = (b   , d, h)
            en.location = (b+.3, d, h)
            empties[name_p] = ep
            empties[name_n] = en

            bone = arm_data.edit_bones.new(f"{h}{d}{b}")
            bone.head = (0,0,0)
            bone.tail = (0,1,0)

    name_p = f"{h}ap"
    name_n = f"{h}an"
    eap = create_empty(name_p)
    ean = create_empty(name_n)
    eap.location = (5  , 0, 0)
    ean.location = (5.3, 0, 0)
    empties[name_p] = eap
    empties[name_n] = ean
    bone = arm_data.edit_bones.new(f"{h}a")
    bone.head = (0,0,0)
    bone.tail = (0,1,0)

bpy.ops.object.mode_set(mode='OBJECT')