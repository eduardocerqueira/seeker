#date: 2024-04-04T16:55:35Z
#url: https://api.github.com/gists/5769a7066842c268bfc62f07171b4eb3
#owner: https://api.github.com/users/kirby561

#
# Replace vertex group with all vertices in a range
#
import bpy
import bmesh

def MakeBoneName(index):
	if index < 10:
		return "Bone.00" + str(index)
	elif index < 100:
		return "Bone.0" + str(index)
	else:
		return "Bone." + str(index)

activeObject = bpy.context.active_object

# Get the mesh data
mesh = activeObject.data

# For each bone
for i in range(102):
	# The vertex group for the bone is the same as its name
	# We named them all "Bone.###" so that's the group name.
	boneVertexGroupName = MakeBoneName(i)
	
	# Find the vertex group
	oldVertexGroup = activeObject.vertex_groups.get(boneVertexGroupName)
	
	# Delete the existing group
	if oldVertexGroup:
		activeObject.vertex_groups.remove(oldVertexGroup)
	
	# Add a fresh one with the same name
	newGroup = activeObject.vertex_groups.new(name=boneVertexGroupName)
	
	# Select the vertices for this bone
	for vert in mesh.vertices:
		minX = i * 0.5 - 0.1
		maxX = i * 0.5 + 0.1
		# Check if the vertex is inside the bounding box
		if (minX <= vert.co.x <= maxX):
			# Add it to the group
			newGroup.add([vert.index], 1.0, 'ADD')
