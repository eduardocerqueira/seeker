#date: 2025-03-21T17:07:00Z
#url: https://api.github.com/gists/dbdd74f6218ba7e44ee687cce400d70c
#owner: https://api.github.com/users/nabesakarenders

import bpy
from shape_keys_plus import core, memory

# Name for the folder shape key; adjust if desired.
FOLDER_NAME = "JCMs"

# Iterate over all selected mesh objects.
for obj in bpy.context.selected_objects:
    if obj.type != 'MESH':
        continue
    if not obj.data.shape_keys:
        continue

    key_blocks = obj.data.shape_keys.key_blocks
    # Skip objects that have only the default "Basic" shape key.
    if len(key_blocks) <= 1:
        continue

    # Set the current object as active (required for core.key.add).
    bpy.context.view_layer.objects.active = obj

    # Create a folder shape key using the add-on's helper.
    folder_key = core.key.add(type='FOLDER')
    folder_key.name = FOLDER_NAME
    # Mark the new shape key as a folder by setting its vertex_group string.
    folder_key.vertex_group = core.folder.generate()

    # Create a tree instance to manage the shape key hierarchy.
    tree_instance = memory.tree()

    # Reparent all shape keys (except "Basic" and the new folder) under the folder.
    for key in key_blocks:
        if key.name in {"Basic", FOLDER_NAME}:
            continue
        tree_instance.transfer(key.name, FOLDER_NAME)

    # Apply the new ordering so that the folder groups its children.
    tree_instance.apply()

    # Auto collapse the folder.
    # Set expand value to 0 using the folder toggle function.
    core.folder.toggle(folder_key, expand=0)

    # Ensure that the "Basic" shape key stays at the top.
    basic_index = key_blocks.find("Basic")
    if basic_index != 0:
        obj.active_shape_key_index = basic_index
        # Move "Basic" upward until it reaches index 0.
        while obj.active_shape_key_index != 0:
            bpy.ops.object.shape_key_move(type='UP')
            basic_index = key_blocks.find("Basic")
            obj.active_shape_key_index = basic_index

    print(f"Processed object '{obj.name}' – grouped shape keys under '{FOLDER_NAME}', with Basic at the top and folder collapsed.")