#date: 2025-02-03T17:05:35Z
#url: https://api.github.com/gists/2e20885c1ec949b01fac371f9ec6af30
#owner: https://api.github.com/users/Alvinislazy

import subprocess
import os
import re

def fetch_output_folder(blender_executable, file):
    """Fetch the output folder from the .blend file using Blender's Python API."""
    try:
        command = [
            blender_executable,
            "-b", file,
            "--python-expr",
            "import bpy; print(bpy.context.scene.render.filepath)"
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output_path = result.stdout.strip()
        return os.path.dirname(output_path)
    except Exception as e:
        return f"Error fetching output folder: {e}"

def open_output_folder(folder):
    """Open the output folder for a file."""
    if os.path.exists(folder):
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])
    else:
        return f"Output folder does not exist: {folder}"