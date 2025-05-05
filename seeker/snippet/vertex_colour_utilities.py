#date: 2025-05-05T16:43:10Z
#url: https://api.github.com/gists/f589fbf1c04d2eef1c43c9cc14fdfe0e
#owner: https://api.github.com/users/Janooba

bl_info = {
    "name":        "Vertex‑Colour Utilities",
    "author":      "Janooba",
    "version":     (1, 0, 0),
    "blender":     (4, 0, 0),
    "description": "Quick paint & colour-based selection tools under Mesh > Vertex Colours",
    "category":    "Mesh",
}

import bpy, bmesh
from math import sqrt

def ensure_corner_byte_layer(mesh):
    for ca in mesh.color_attributes:
        if ca.domain == 'CORNER' and ca.data_type == 'BYTE_COLOR':
            mesh.color_attributes.active_color = ca
            return ca
    ca = mesh.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
    mesh.color_attributes.active_color = ca
    return ca

def bm_color_layer(bm, layer_name):
    """Return (or create) a loop colour layer handle on the BMesh."""
    layer = bm.loops.layers.color.get(layer_name)
    if not layer:
        layer = bm.loops.layers.color.new(layer_name)
    return layer

class MESH_OT_vc_paint_selected(bpy.types.Operator):
    bl_idname  = "mesh.vc_paint_selected"
    bl_label   = "Paint Selected Faces…"
    bl_options = {'REGISTER', 'UNDO'}

    # live-preview colour
    def _update_color(self, ctx):
        self._apply(ctx)

    color: bpy.props.FloatVectorProperty(
        name="Colour", subtype='COLOR', size=4,
        min=0, max=1, default=(1., 1., 1., 1.),
        update=_update_color)

    # ------------------------------------------------ #
    #  Single paint routine
    # ------------------------------------------------ #
    def _apply(self, ctx):
        obj, mesh = ctx.active_object, ctx.active_object.data
        ca = ensure_corner_byte_layer(mesh)

        bm   = bmesh.from_edit_mesh(mesh)
        layer = bm.loops.layers.color.get(ca.name)

        for f in bm.faces:
            if f.select:
                for lp in f.loops:
                    lp[layer] = self.color

        bmesh.update_edit_mesh(mesh, destructive=True)

    # ------------------------------------------------ #
    #  Boilerplate
    # ------------------------------------------------ #
    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o and o.type == 'MESH' and ctx.mode == 'EDIT_MESH'

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self, width=220)

    def execute(self, ctx):
        self._apply(ctx)
        return {'FINISHED'}

    def draw(self, _):
        self.layout.prop(self, "color")

class MESH_OT_vc_select_by_colour(bpy.types.Operator):
    bl_idname  = "mesh.vc_select_by_colour"
    bl_label   = "Select Faces by Colour…"
    bl_options = {'REGISTER', 'UNDO'}

    reference: bpy.props.FloatVectorProperty(name="Reference Colour",
                                             subtype='COLOR',
                                             size=3, min=0, max=1)
    tolerance: bpy.props.FloatProperty(name="Tolerance",
                                       min=0, max=1, default=0.05,
                                       description="Max RGB distance")

    # helpers
    def _average_colour(self, face, layer):
        rgb = [0, 0, 0]
        for lp in face.loops:
            c = lp[layer][:3]
            rgb[0] += c[0]; rgb[1] += c[1]; rgb[2] += c[2]
        l = len(face.loops)
        return (rgb[0]/l, rgb[1]/l, rgb[2]/l)

    # operator flow
    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o and o.type == 'MESH' and ctx.mode == 'EDIT_MESH'

    def invoke(self, ctx, _evt):
        # default reference = average of current selection (or white)
        mesh = ctx.active_object.data
        ca   = ensure_corner_byte_layer(mesh)
        bm   = bmesh.from_edit_mesh(mesh)
        layer = bm_color_layer(bm, ca.name)

        sel_cols = [ self._average_colour(f, layer) for f in bm.faces if f.select ]
        if sel_cols:
            avg = [ sum(ch)/len(sel_cols) for ch in zip(*sel_cols) ]
            self.reference = avg
        else:
            self.reference = (1, 1, 1)
        return ctx.window_manager.invoke_props_dialog(self, width=220)

    def draw(self, _ctx):
        col = self.layout.column()
        col.prop(self, "reference", text="")
        col.prop(self, "tolerance")

    def execute(self, _ctx):
        mesh = bpy.context.active_object.data
        ca   = ensure_corner_byte_layer(mesh)
        bm   = bmesh.from_edit_mesh(mesh)
        layer = bm_color_layer(bm, ca.name)

        ref = self.reference
        tol = self.tolerance

        def dist(c):
            return sqrt((c[0]-ref[0])**2 + (c[1]-ref[1])**2 + (c[2]-ref[2])**2)

        for f in bm.faces:
            c = self._average_colour(f, layer)
            f.select = dist(c) <= tol

        bmesh.update_edit_mesh(mesh, destructive=False, loop_triangles=False)
        return {'FINISHED'}

#  Sub-menu & registration glue
class VIEW3D_MT_mesh_vertex_colours(bpy.types.Menu):
    bl_label = "Vertex Colours"

    def draw(self, ctx):
        layout = self.layout
        layout.operator("mesh.vc_paint_selected", icon='BRUSH_DATA')
        layout.operator("mesh.vc_select_by_colour", icon='COLOR')

def menu_entry(self, _):
    self.layout.menu("VIEW3D_MT_mesh_vertex_colours", icon='COLOR')

CLASSES = (
    MESH_OT_vc_paint_selected,
    MESH_OT_vc_select_by_colour,
    VIEW3D_MT_mesh_vertex_colours,
)

def register():
    for c in CLASSES: bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_edit_mesh.append(menu_entry)

def unregister():
    bpy.types.VIEW3D_MT_edit_mesh.remove(menu_entry)
    for c in reversed(CLASSES): bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()