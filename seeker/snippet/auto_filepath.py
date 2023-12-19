#date: 2023-12-19T16:54:06Z
#url: https://api.github.com/gists/c56c5f4bf1e42238928bf2cef481cde0
#owner: https://api.github.com/users/QuietNoise

# Inspired by original code by Robert Guetzkow: https://gist.github.com/robertguetzkow/8dacd4b565538d657b72efcaf0afe07e
# If you want save still image renders (F12) you must have at least one File Output node connected to image output in your compositor.
# The node's name will be prepended to the filename as well as the timestamp.
# The timestamp is created when the job is started and not when the frame is rendered thus for animation renders the timestamp
# will be the same for all frames.
# There is also an option to put entire animation renders in its own timestampted subfolder.


bl_info = {
    "name": "Auto-Filename",
    "author": "QuietNoise",
    "version": (1, 0, 1),
    "blender": (4, 00, 0),
    "location": "Output Properties > Auto-filename",
    "description": "Automatically sets a unique filename for each frame based on the current timestamp. Inspired by original code by Robert Guetzkow",
    "warning": "",
    "wiki_url": "",
    "category": "Render"}

import bpy
import datetime
from pathlib import Path
from bpy.app.handlers import persistent


@persistent
def update_filename(self):
    # If auto filename generation is disabled, do nothing.
    if not bpy.context.scene.auto_filename_settings.use_auto_filename:
        return

    # Get the current timestamp and Path object for the base directory.
    now = datetime.datetime.now()
    base_path = Path(bpy.context.scene.auto_filename_settings.directory)
    foldername = now.strftime('%Y-%m-%d_%H_%M_%S_%f')
    filename = foldername + " - ####"

    # Should animation renders be put in a subfolder?
    if bpy.context.scene.auto_filename_settings.animations_in_subfolder:
        bpy.context.scene.render.filepath = str(base_path / foldername / filename)
    else:
        bpy.context.scene.render.filepath = str(base_path / filename )

    if bpy.context.scene.use_nodes:
        for node in bpy.context.scene.node_tree.nodes:
            if node.type == "OUTPUT_FILE":
                node.file_slots[0].path = node.name + " - " + filename


def set_directory(self, value):
    path = Path(value)
    if path.is_dir():
        self["directory"] = value


def get_directory(self):
    return self.get("directory", bpy.context.scene.auto_filename_settings.bl_rna.properties["directory"].default)


class AutoFilenameSettings(bpy.types.PropertyGroup):
    use_auto_filename: bpy.props.BoolProperty(name="Automatic filename generation.",
                                              description="Enable/disable automatic filename generation for renders",
                                              default=False)

    directory: bpy.props.StringProperty(name="Directory",
                                        description="Directory where files shall be stored",
                                        default="/",
                                        maxlen=4096,
                                        subtype="DIR_PATH",
                                        set=set_directory,
                                        get=get_directory)

    animations_in_subfolder: bpy.props.BoolProperty(name="Put animation jobs in subfolder.",
                                              description="Whether animation render jobs should be put in a timestamped subfolder",
                                              default=False)


class AUTOFILENAME_PT_panel(bpy.types.Panel):
    bl_label = "Auto-Filename"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "output"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.auto_filename_settings, "use_auto_filename", text="")

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene.auto_filename_settings, "directory")
        layout.prop(context.scene.auto_filename_settings, "animations_in_subfolder", text="Put animation jobs in subfolder")


classes = (AutoFilenameSettings, AUTOFILENAME_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.auto_filename_settings = bpy.props.PointerProperty(type=AutoFilenameSettings)
    if update_filename not in bpy.app.handlers.render_init:
        bpy.app.handlers.render_init.append(update_filename)



def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.auto_filename_settings
    if update_filename in bpy.app.handlers.render_init:
        bpy.app.handlers.render_init.remove(update_filename)


if __name__ == "__main__":
    register()