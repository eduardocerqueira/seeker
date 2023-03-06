#date: 2023-03-06T17:06:58Z
#url: https://api.github.com/gists/56b557d632f7a055d8742064fe489478
#owner: https://api.github.com/users/SmittyWerbenJJ

""" 
This one is a naiive approach on trying to fix the mix rgb nodes, when opening blender 3.4+ in blender 3.3
The broken nodes shall be replaced with the default mixrgb node while keeping original connections 
Code may contain lots of bugs.

To use: 
  1.open/drag the file in the a blender text editor
  2.go to shader editor
  3.right click anywhere on graph and select "Replace Undefined MixRGB Nodes"
"""

import bpy


def replace_nodes():
    node_tree = bpy.context.active_object.active_material.node_tree
    nodes = node_tree.nodes

    for node in nodes:
        # only consider undefined nodes
        if node.bl_idname != "NodeUndefined":
            continue

        input_names = [input.name for input in node.inputs]
        if not all(name in input_names for name in ["Factor", "A", "B"]):
            continue

        # set the output link
        replace_one_MixRGBNode(node_tree, node)

        # # for i,outputsocket in enumerate(node.outputs):
        # #     links_table[mixrgb_node.inputs[i]]=outputsocket.links[0].to_socket

        # for _from,_to in links_table.items():

        # for input_name, mix_input_name in [("Factor", "Fac"), ("A", "Color1"), ("B", "Color2")]:
        #     if mixrgb_node.inputs[mix_input_name].bl_idname== "NodeSocketFloatFactor"   :
        #         mixrgb_node.inputs[mix_input_name].default_value = node.inputs[input_name].default_value
        #     elif mixrgb_node.inputs[mix_input_name].bl_idname== "NodeSocketVector"   :
        #         mixrgb_node.inputs[mix_input_name].default_value = [0.0, 0.0, 0.0, 1.0]

        # nodes.remove(node)


def replace_one_MixRGBNode(
    node_tree: bpy.types.NodeTree, brokenNode: bpy.types.Node
):
    from_sockets = []
    to_sockets = []
    input_links = []
    output_links = []
    link_template = {"from_socket": None, "to_socket": None}
    remappedSockets = {
        "Factor": "Fac",
        "A": "Color1",
        "B": "Color2",
        "Result": "Color",
    }

    broken_input_names = list(remappedSockets.keys())[:3]
    broken_output_name = list(remappedSockets.keys())[-1]
    mixrgb_node = node_tree.nodes.new(type="ShaderNodeMixRGB")
    mixrgb_node.location = brokenNode.location

    for nodeInput in brokenNode.inputs:
        if nodeInput.name not in broken_input_names:
            continue

        for link in nodeInput.links:
            link: bpy.types.NodeLink
            if link.to_socket.name not in list(remappedSockets.keys()):
                continue
            newLink = link_template.copy()
            newLink["from_socket"] = link.from_socket
            newLink["to_socket"] = mixrgb_node.inputs[
                remappedSockets[link.to_socket.name]
            ]
            input_links.append(newLink)

    for nodeOutput in brokenNode.outputs:
        if not nodeOutput.name == broken_output_name:
            continue

        for link in nodeOutput.links:
            link: bpy.types.NodeLink
            if link.from_socket.name not in list(remappedSockets.keys()):
                continue
            newLink = link_template.copy()
            newLink["from_socket"] = mixrgb_node.outputs[
                remappedSockets[link.from_socket.name]
            ]
            newLink["to_socket"] = link.to_socket
            output_links.append(newLink)

    node_tree.nodes.remove(brokenNode)

    for thelink in input_links + output_links:
        _from = thelink["from_socket"]
        _to = thelink["to_socket"]
        print(f"connecting {str(_from)}  to {str(_to)} ")
        node_tree.links.new(_from, _to)

    # for thelink in output_links:
    #     _from=mixrgb_node.outputs[0]
    #     _to=socket
    #     print(f"connecting {_from}  to {_to} ")
    #     node_tree.links.new( _from,_to)


# Define the function to add a button to the Shader Editor
def add_replace_button(self, context):
    layout = self.layout
    layout.operator(
        "shader.replace_undefined_nodes", text="Replace Undefined MixRGB Nodes"
    )


# Define the operator to call the replace function
class ReplaceUndefinedNodes(bpy.types.Operator):
    bl_idname = "shader.replace_undefined_nodes"
    bl_label = "Replace Undefined Nodes"

    def execute(self, context):
        replace_nodes()
        return {"FINISHED"}


# Register the operator and add the button to the Shader Editor
def register():
    bpy.utils.register_class(ReplaceUndefinedNodes)
    bpy.types.NODE_MT_context_menu.append(add_replace_button)


# Unregister the operator and remove the button from the Shader Editor
def unregister():
    bpy.types.NODE_MT_context_menu.remove(add_replace_button)
    bpy.utils.unregister_class(ReplaceUndefinedNodes)


# Run the registration functions
if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    register()
