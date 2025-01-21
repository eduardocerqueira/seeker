#date: 2025-01-21T17:02:16Z
#url: https://api.github.com/gists/e0442c87e8b9bd65283d0fd356113412
#owner: https://api.github.com/users/rnielikki

# Refactoring? Meh
# I learnt Python JUST FOR THIS, Also this is my first Blender coding ever
# Idk how to define function
# Zero addon generation, I don't have it D:<
# ALL HAND WRITTEN AAAAA
# Anyway Enjoy
#
# How to:
# Open default Blender project, go to python console, and Ctrl+C Ctrl+V

a3d = [area for area in bpy.context.screen.areas if area.type == 'VIEW_3D'][0]
r3d = a3d.spaces[0].region_3d
r3d.view_distance = 1.4
r3d.view_location.z = -0.45

for ob in bpy.context.scene.objects:
   if ob.type == "MESH":
       ob.select_set(True)
       bpy.ops.object.delete()

bpy.ops.mesh.primitive_torus_add(major_segments=24, minor_segments=12, major_radius=0.12, minor_radius=0.06)
mesh_bread = bpy.context.view_layer.objects.active

# Must go edit mode first and deselect all to work properly
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')

# must done in object mode first in object mode
obj = bpy.context.object
for v in obj.data.vertices:
   if v.co.z >= 0:
      obj.data.vertices[v.index].select = True

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.duplicate()
bpy.ops.mesh.separate(type='SELECTED')
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.object.shade_smooth()

mesh_bread.select_set(False)
bpy.context.view_layer.objects.active = bpy.context.view_layer.objects.selected[0]

mesh_coat = bpy.context.view_layer.objects.active

mesh_bread.modifiers.new("vertex gore", "SUBSURF")

bpy.data.objects[mesh_bread.name].select_set(False)
bpy.context.view_layer.objects.active = bpy.data.objects[mesh_coat.name]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')
obj = bpy.context.object
for v in obj.data.vertices:
   if v.co.z == 0:
      obj.data.vertices[v.index].select = True

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.transform.translate(value=(0.0,0.0,-0.01))
bpy.ops.mesh.select_nth(skip=1, nth=1)
bpy.ops.transform.translate(value=(0.0,0.0,0.02))

bpy.ops.object.mode_set(mode='OBJECT')
md = mesh_coat.modifiers.new("melt", "SIMPLE_DEFORM")
md.deform_method = 'TWIST'
md.angle=-0.5;
mesh_coat.modifiers.new("vertex gore", "SUBSURF")
md = mesh_coat.modifiers.new("phlat", "SHRINKWRAP")
md.target = mesh_bread
md = mesh_coat.modifiers.new("coat", "SOLIDIFY")
md.offset = 1
md.thickness = 0.005

bread_mat = bpy.data.materials.new("bread")
coat_mat = bpy.data.materials.new("sugar")
mesh_bread.data.materials.append(bread_mat)
mesh_coat.data.materials.append(coat_mat)
bread_mat.diffuse_color = (0.8, 0.5, 0.2, 1)
coat_mat.diffuse_color = (0.8, 0.3, 0.7, 1)
coat_mat.roughness = 0.25
sprinkle_mat = bpy.data.materials.new("sprinkle")

bpy.ops.node.new_geometry_nodes_modifier()
sugar_surf = bpy.data.node_groups["Geometry Nodes"]
gm_input = sugar_surf.nodes["Group Input"]
gm_output = sugar_surf.nodes["Group Output"]
gm_join = sugar_surf.nodes.new("GeometryNodeJoinGeometry")
gm_inst = sugar_surf.nodes.new("GeometryNodeInstanceOnPoints")
gm_pf = sugar_surf.nodes.new("GeometryNodeDistributePointsOnFaces")
gm_uv = sugar_surf.nodes.new("GeometryNodeMeshCylinder")
gm_mat = sugar_surf.nodes.new("GeometryNodeSetMaterial")

gm_math_mul = sugar_surf.nodes.new("ShaderNodeMath")
gm_math_rot = sugar_surf.nodes.new("ShaderNodeMath")
gm_math_gt = sugar_surf.nodes.new("ShaderNodeMath")
gm_sep = sugar_surf.nodes.new("ShaderNodeSeparateXYZ")
gm_norm = sugar_surf.nodes.new("GeometryNodeInputNormal")

gm_math_mul.operation = "MULTIPLY"
gm_math_rot.operation = "MULTIPLY"
gm_math_gt.operation = "GREATER_THAN"
gm_math_rot.inputs[1].default_value = 180
gm_math_mul.inputs[1].default_value = 2500
gm_math_gt.inputs[1].default_value = 0.4
gm_mat.inputs["Material"].default_value = sprinkle_mat

gm_uv.inputs["Vertices"].default_value=8
gm_uv.inputs["Radius"].default_value=0.002
gm_uv.inputs["Depth"].default_value=0.01

sugar_surf.links.new(gm_join.outputs["Geometry"], gm_output.inputs["Geometry"])
sugar_surf.links.new(gm_input.outputs["Geometry"], gm_join.inputs["Geometry"])
sugar_surf.links.new(gm_inst.outputs["Instances"], gm_join.inputs["Geometry"])
sugar_surf.links.new(gm_pf.outputs["Points"], gm_inst.inputs["Points"])
sugar_surf.links.new(gm_uv.outputs["Mesh"], gm_mat.inputs["Geometry"])
sugar_surf.links.new(gm_mat.outputs["Geometry"], gm_inst.inputs["Instance"])
sugar_surf.links.new(gm_input.outputs["Geometry"], gm_pf.inputs["Mesh"])

sugar_surf.links.new(gm_inst.inputs["Rotation"], gm_math_rot.outputs["Value"])
sugar_surf.links.new(gm_math_rot.inputs[0], gm_pf.outputs["Normal"])

sugar_surf.links.new(gm_pf.inputs["Selection"], gm_math_gt.outputs["Value"])
sugar_surf.links.new(gm_math_gt.inputs[0], gm_sep.outputs["Z"])
sugar_surf.links.new(gm_sep.inputs["Vector"], gm_norm.outputs["Normal"])
sugar_surf.links.new(gm_pf.inputs["Density"], gm_math_mul.outputs["Value"])
sugar_surf.links.new(gm_math_mul.inputs[0], gm_sep.outputs["Z"])

sprinkle_mat.use_nodes = True
shader_root = sprinkle_mat.node_tree.nodes["Principled BSDF"]
shader_obj = sprinkle_mat.node_tree.nodes.new("ShaderNodeObjectInfo")
shader_clr = sprinkle_mat.node_tree.nodes.new("ShaderNodeValToRGB")
shader_clr.color_ramp.elements[0].color = (1.0, 0.0, 0.0, 1.0)
shader_clr.color_ramp.elements.new(0.250)
shader_clr.color_ramp.elements[1].color = (1.0, 1.0, 0.0, 1.0)
shader_clr.color_ramp.elements.new(0.5)
shader_clr.color_ramp.elements[2].color = (0.0, 1.0, 0.0, 1.0)
shader_clr.color_ramp.elements.new(0.75)
shader_clr.color_ramp.elements[3].color = (0.0, 0.0, 1.0, 1.0)
shader_clr.color_ramp.elements[4].color = (1.0, 1.0, 1.0, 1.0)

sprinkle_mat.node_tree.links.new(shader_root.inputs["Base Color"], shader_clr.outputs["Color"])
sprinkle_mat.node_tree.links.new(shader_clr.inputs["Fac"], shader_obj.outputs["Random"])

bpy.context.scene.camera = [x for x in bpy.context.scene.objects if x.type == 'CAMERA'][0]

r3d.view_distance = 1.8

rg = [region for region in a3d.regions if region.type == 'WINDOW'][0]
with bpy.context.temp_override(window=bpy.context.window, area=a3d, region = rg):
   bpy.ops.view3d.camera_to_view()

bpy.context.scene.eevee.use_shadows = False

bpy.ops.render.render('INVOKE_DEFAULT')