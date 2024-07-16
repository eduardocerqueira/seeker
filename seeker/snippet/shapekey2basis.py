#date: 2024-07-16T17:11:02Z
#url: https://api.github.com/gists/83a130c233d7c876303a0b78b6970b25
#owner: https://api.github.com/users/Ximmer-VR

# MIT License
#
# Copyright (c) 2017 GiveMeAllYourCats
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Code author: Hotox
# Repo: https://github.com/michaeldegroot/cats-blender-plugin
# Edits by: Ximmer
#
# Portions taken from tools/shapekey.py
#

bl_info = {
    "name": "Shape Key to Basis",
    "blender": (3, 6, 0),
    "category": "Object",
}

import bpy

# Define a simple operator
class OBJECT_OT_apply_selected_shapekey_to_basis(bpy.types.Operator):
    bl_idname = "object.apply_selected_shapekey_to_basis"
    bl_label = "Apply Selected Shapekey to Basis"

    def error(self, msg):
        self.report({'ERROR'}, str(msg))

    def get_active(self):
        return bpy.context.view_layer.objects.active

    def curr_mode(self):
        return self.get_active().mode

    def switch(self, mode):
        if self.get_active().mode == mode:
            return
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode=mode, toggle=False)

    def has_shapekeys(self, mesh):
        if not hasattr(mesh.data, 'shape_keys'):
            return False
        return hasattr(mesh.data.shape_keys, 'key_blocks')

    def sort_shape_keys(self, mesh_name, shape_key_order=None):
        mesh = bpy.context.view_layer.objects[mesh_name]
        if not self.has_shapekeys(mesh):
            return
        bpy.context.view_layer.objects.active = mesh

        if not shape_key_order:
            shape_key_order = []

        order = [
            'Basis',
            'Basis Original'
        ]

        for shape in shape_key_order:
            if shape not in order:
                order.append(shape)

        wm = bpy.context.window_manager
        current_step = 0
        wm.progress_begin(current_step, len(order))

        i = 0
        for name in order:
            if name == 'Basis' and 'Basis' not in mesh.data.shape_keys.key_blocks:
                i += 1
                current_step += 1
                wm.progress_update(current_step)
                continue

            for index, shapekey in enumerate(mesh.data.shape_keys.key_blocks):
                if shapekey.name == name:

                    mesh.active_shape_key_index = index
                    new_index = i
                    index_diff = (index - new_index)

                    if new_index >= len(mesh.data.shape_keys.key_blocks):
                        bpy.ops.object.shape_key_move(type='BOTTOM')
                        break

                    position_correct = False
                    if 0 <= index_diff <= (new_index - 1):
                        while position_correct is False:
                            if mesh.active_shape_key_index != new_index:
                                bpy.ops.object.shape_key_move(type='UP')
                            else:
                                position_correct = True
                    else:
                        if mesh.active_shape_key_index > new_index:
                            bpy.ops.object.shape_key_move(type='TOP')

                        position_correct = False
                        while position_correct is False:
                            if mesh.active_shape_key_index != new_index:
                                bpy.ops.object.shape_key_move(type='DOWN')
                            else:
                                position_correct = True

                    i += 1
                    break

            current_step += 1
            wm.progress_update(current_step)

        mesh.active_shape_key_index = 0

        wm.progress_end()

    def execute(self, context):

        curr_mode = self.curr_mode()

        self.switch('OBJECT')

        mesh = self.get_active()

        # Get shapekey which will be the new basis
        new_basis_shapekey = mesh.active_shape_key
        new_basis_shapekey_name = new_basis_shapekey.name
        new_basis_shapekey_value = new_basis_shapekey.value

        # Check for reverted shape keys
        if ' - Reverted' in new_basis_shapekey_name and new_basis_shapekey.relative_key.name != 'Basis':
            for shapekey in mesh.data.shape_keys.key_blocks:
                if ' - Reverted' in shapekey.name and shapekey.relative_key.name == 'Basis':
                    self.error('todo')
                    #Common.show_error(t('ShapeKeyApplier.error.revert.scale'), t('ShapeKeyApplier.error.revert', name=shapekey.name))
                    return {'FINISHED'}

            self.error('todo')
            #Common.show_error(t('ShapeKeyApplier.error.revert.scale'), t('ShapeKeyApplier.error.revert'))
            return {'FINISHED'}

        # Set up shape keys
        mesh.show_only_shape_key = False
        bpy.ops.object.shape_key_clear()

        # Create a copy of the new basis shapekey to make it's current value stay as it is
        new_basis_shapekey.value = new_basis_shapekey_value
        if new_basis_shapekey_value == 0:
            new_basis_shapekey.value = 1
        new_basis_shapekey.name = new_basis_shapekey_name + '--Old'

        # Replace old new basis with new new basis
        new_basis_shapekey = mesh.shape_key_add(name=new_basis_shapekey_name, from_mix=True)
        new_basis_shapekey.value = 1

        # Delete the old one
        for index in reversed(range(0, len(mesh.data.shape_keys.key_blocks))):
            mesh.active_shape_key_index = index
            shapekey = mesh.active_shape_key
            if shapekey.name == new_basis_shapekey_name + '--Old':
                bpy.ops.object.shape_key_remove(all=False)
                break

        # Find old basis and rename it
        old_basis_shapekey = mesh.data.shape_keys.key_blocks[0]
        old_basis_shapekey.name = new_basis_shapekey_name + ' - Reverted'
        old_basis_shapekey.relative_key = new_basis_shapekey

        # Rename new basis after old basis was renamed
        new_basis_shapekey.name = 'Basis'

        # Mix every shape keys with the new basis
        for index in range(0, len(mesh.data.shape_keys.key_blocks)):
            mesh.active_shape_key_index = index
            shapekey = mesh.active_shape_key
            if shapekey and shapekey.name != 'Basis' and ' - Reverted' not in shapekey.name:
                shapekey.value = 1
                mesh.shape_key_add(name=shapekey.name + '-New', from_mix=True)
                shapekey.value = 0

        # Remove all the unmixed shape keys except basis and the reverted ones
        for index in reversed(range(0, len(mesh.data.shape_keys.key_blocks))):
            mesh.active_shape_key_index = index
            shapekey = mesh.active_shape_key
            if shapekey and not shapekey.name.endswith('-New') and shapekey.name != 'Basis' and ' - Reverted' not in shapekey.name:
                bpy.ops.object.shape_key_remove(all=False)

        # Fix the names and the relative shape key
        for index, shapekey in enumerate(mesh.data.shape_keys.key_blocks):
            if shapekey and shapekey.name.endswith('-New'):
                shapekey.name = shapekey.name[:-4]
                shapekey.relative_key = new_basis_shapekey

        self.sort_shape_keys(mesh.name)

        # Correctly apply the new basis as basis (important step, doesn't work otherwise)
        self.switch('EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.remove_doubles(threshold=0)
        self.switch('OBJECT')

        # If a reversed shapekey was applied as basis, fix the name
        if ' - Reverted - Reverted' in old_basis_shapekey.name:
            old_basis_shapekey.name = old_basis_shapekey.name.replace(' - Reverted - Reverted', '')
            self.report({'INFO'}, 'Successfully removed shapekey ""{name}"" from the Basis.'.format(name=old_basis_shapekey.name))
        else:
            self.report({'INFO'}, 'Successfully set shapekey ""{name}"" as the new Basis.'.format(name=new_basis_shapekey_name))

        self.switch(curr_mode)

        return {'FINISHED'}

# Define a function to draw the menu item
def draw_item(self, context):
    layout = self.layout
    layout.operator(OBJECT_OT_apply_selected_shapekey_to_basis.bl_idname, text="Apply Selected Shapekey to Basis")

# Register the operator and the menu item
def register():
    bpy.utils.register_class(OBJECT_OT_apply_selected_shapekey_to_basis)
    bpy.types.MESH_MT_shape_key_context_menu.append(draw_item)

def unregister():
    bpy.types.MESH_MT_shape_key_context_menu.remove(draw_item)
    bpy.utils.unregister_class(OBJECT_OT_apply_selected_shapekey_to_basis)

if __name__ == "__main__":
    register()
