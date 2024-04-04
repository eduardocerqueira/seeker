#date: 2024-04-04T16:55:35Z
#url: https://api.github.com/gists/5769a7066842c268bfc62f07171b4eb3
#owner: https://api.github.com/users/kirby561

#
# Create list of bones all parented to a single parent bone at algorithmic start/end location:
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

# Create a new armature object
armature = bpy.data.armatures.new("Armature")
armature_obj = bpy.data.objects.new("Armature", armature)

# Add to scene and select it
bpy.context.collection.objects.link(armature_obj)
bpy.context.view_layer.objects.active = armature_obj

# We need to be in edit mode to edit the bones of the armature
bpy.ops.object.mode_set(mode='EDIT')

# make a single parent bone
parentBone = armature_obj.data.edit_bones.new("ParentBone")
parentBone.head = (0, 0, 0)
parentBone.tail = (0.5, 0, 0)

# Create bones
for i in range(102):
	bone = armature_obj.data.edit_bones.new(MakeBoneName(i))
	bone.head = (i * 0.5, 0, 0)
	bone.tail = ((i + 1)*0.5, 0, 0)
	bone.parent = parentBone
  