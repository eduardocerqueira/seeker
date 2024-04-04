#date: 2024-04-04T16:55:35Z
#url: https://api.github.com/gists/5769a7066842c268bfc62f07171b4eb3
#owner: https://api.github.com/users/kirby561

#
# Extrude selected faces N times
#
import bpy
import bmesh

for i in range(99):
	bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(.5, 0, 0)})
