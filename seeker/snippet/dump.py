#date: 2025-01-31T17:04:37Z
#url: https://api.github.com/gists/6e03b7fc14b82a15016b3d3756b1fa42
#owner: https://api.github.com/users/fdciabdul

import json
import logging
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Frame, Label
from prompt_toolkit.formatted_text import FormattedText
from poco.drivers.android.uiautomation import AndroidUiautomationPoco

logging.getLogger("airtest").setLevel(logging.WARNING)
logging.getLogger("poco").setLevel(logging.WARNING)
logging.getLogger("airtest.core.android.adb").setLevel(logging.WARNING)

poco = AndroidUiautomationPoco()

ui_tree = poco.dump()

class TreeNode:
    def __init__(self, name, children=None, expanded=True):  # Default expanded=True
        self.name = name
        self.children = children or []
        self.expanded = expanded

    def toggle(self):
        self.expanded = not self.expanded


def build_tree(data):
    name = data.get("name", f"Unnamed({id(data)})")
    children = [build_tree(child) for child in data.get("children", [])]
    return TreeNode(name, children)


root_node = build_tree(ui_tree)
selected_index = 0


def render_tree(node, indent=0, selected=False):
    prefix = "  " * indent
    marker = "[+] " if node.expanded else "[ ] "
    color = "bold cyan" if selected else "white"

    text = [(color, f"{prefix}{marker}{node.name}\n")] 

    if node.expanded:
        for child in node.children:
            text.extend(render_tree(child, indent + 1, selected=False))

    return text  


def get_flattened_nodes(node, nodes=None):
    if nodes is None:
        nodes = []
    nodes.append(node)
    if node.expanded:
        for child in node.children:
            get_flattened_nodes(child, nodes)
    return nodes

tree_label = Label("")

def update_ui():
    nodes = get_flattened_nodes(root_node)
    formatted_text = FormattedText(
        [segment for i, node in enumerate(nodes) for segment in render_tree(node, indent=i, selected=(i == selected_index))]
    )
    tree_label.text = formatted_text 

kb = KeyBindings()

@kb.add("up")
def _(event):
    global selected_index
    if selected_index > 0:
        selected_index -= 1
        update_ui()
        event.app.invalidate()  

@kb.add("down")
def _(event):
    global selected_index
    if selected_index < len(get_flattened_nodes(root_node)) - 1:
        selected_index += 1
        update_ui()
        event.app.invalidate()  

@kb.add("enter")
def _(event):
    global selected_index
    nodes = get_flattened_nodes(root_node)
    if 0 <= selected_index < len(nodes):
        nodes[selected_index].toggle()
        update_ui()
        event.app.invalidate()  

@kb.add("q")
def _(event):
    "Quit application"
    event.app.exit()


update_ui() 
frame = Frame(tree_label, title="UI Hierarchy Explorer")

layout = Layout(frame)
app = Application(layout=layout, key_bindings=kb, full_screen=True)

app.run()
