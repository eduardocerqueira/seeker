#date: 2022-07-19T17:17:22Z
#url: https://api.github.com/gists/b85e33b7c67a3705e663b2e446984c39
#owner: https://api.github.com/users/Modder4869

import bpy
# select all empty objects
bpy.ops.object.select_by_type(type="EMPTY")
# delete them
bpy.ops.object.delete()
# set body to active 
bpy.context.view_layer.objects.active = bpy.data.objects["Body"]
#select body
bpy.data.objects["Body"].select_set(state=True, view_layer=None)
meshes = ["Body", "EyeStar", "EffectMesh"]
# select anything thats not Body , EyeStar,EffectMesh
for object in bpy.context.scene.objects:
    if not object.name in meshes and object.type == "MESH":
        object.select_set(state=True, view_layer=None)
        #join mesh
bpy.ops.object.join()
# append blender file
with bpy.data.libraries.load(
    "D:\Xodd\Blender-miHoYo-Shaders\miHoYo - Genshin Impact.blend"
) as (data_from, data_to):
    data_to.materials = data_from.materials
    # link materials 
for object in bpy.data.objects:
    name = object.name
    if "Body" in name:
        for materialSlot in object.material_slots:
            if "Body" in materialSlot.name:
                materialSlot.material = bpy.data.materials["miHoYo - Genshin Body"]
            elif "Face" in materialSlot.name:
                materialSlot.material = bpy.data.materials["miHoYo - Genshin Face"]
            elif "Hair" in materialSlot.name:
                materialSlot.material = bpy.data.materials["miHoYo - Genshin Hair"]
            elif "Dress" in materialSlot.name:#is it Body or Hair i have no idea
                materialSlot.material = bpy.data.materials["miHoYo - Genshin Hair"]

    else:
        pass
    #hides EyeStar and EffectMesh
bpy.data.objects['EffectMesh'].hide_set(True);
bpy.data.objects['EyeStar'].hide_set(True);
#run genshin texture import script
bpy.ops.script.python_file_run(filepath="D:\Xodd\Blender-miHoYo-Shaders\scripts\genshin-import-linear.py")
#set view transform to standard
bpy.context.scene.view_settings.view_transform = 'Standard'