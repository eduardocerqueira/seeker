#date: 2022-07-08T17:11:03Z
#url: https://api.github.com/gists/553069aca94106e51cb4d5b646582ad1
#owner: https://api.github.com/users/BelzarSirus

# Sets object to smouth and set auto normal and angle of 180 degress

# Location to put the file, with difrent name or version
# C:\Users\Belzar\AppData\Roaming\Blender Foundation\Blender\3.2\scripts\addons


import bpy


bl_info = {
    "name": "Belzar Auto Normal",
    "version": (1, 0),
    "author": "Belzar Sirus",
    "blender": (3, 2, 0),
    "description": "Turn on auto normal and st to 180",
    "category": "Object",
}


class OBJECT_OT_belz_set_norm(bpy.types.Operator):
    """ToolTip"""
    bl_idname = "object.belz_setnorm"
    bl_label = "Turn on auto normal and st to 180"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        bpy.ops.object.shade_smooth()
        for o in bpy.context.selected_objects:
            o.data.use_auto_smooth = True
            o.data.auto_smooth_angle = 3.14159

        return {'FINISHED'}


def Belz_Set_Norm_Menu_Draw(self, context):
    self.layout.operator('object.belz_setnorm',
                         text='Set Normals')


def register():
    bpy.utils.register_class(OBJECT_OT_belz_set_norm)
    bpy.types.VIEW3D_MT_object_context_menu.append(Belz_Set_Norm_Menu_Draw)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_belz_set_norm)
    bpy.types.VIEW3D_MT_object_context_menu.remove(Belz_Set_Norm_Menu_Draw)


if __name__ == '__main__':
    register()
