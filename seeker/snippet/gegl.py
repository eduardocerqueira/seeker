#date: 2025-10-24T17:13:03Z
#url: https://api.github.com/gists/c88a4e263ad8f0bd596b7f8c2938cd5c
#owner: https://api.github.com/users/CodeZombie

from PIL import Image
import torch
import numpy as np
import tempfile
import subprocess
import time

# NOTE: Ensure this is actually removing temp files.
# NOTE: Check to see if this works on linux.

"""
Runs an image through a GEGL operation.

You must provide a path to a gegl executable. You can find this in your gimp install path. You do have gimp installed, right?

I don't really have any tips on how to write gegl commands.
I've been using gegl on and off for 5 years and I still don't know how to use it because their documentation is borderline non-existent.
"""

class GEGL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gegl_path": ("STRING", {
                    "default": r"C:\Program Files\GIMP 3\bin\gegl.exe", }),
                "include_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, the alpha channel will be included in the output image."}),
                "operation": ("STRING", {
                    "default": "slic cluster-size=76 compactness=2 iterations=8", 
                    "multiline": True,
                    "values": ["slic cluster-size=76 compactness=2 iterations=8"]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gegl"

    CATEGORY = "image/Jeremy Nodes"

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def gegl(self, image, gegl_path, include_alpha, operation):
        pil_image = self.tensor2pil(image).convert("RGBA")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as input_file:
            pil_image.save(input_file.name)
            output_file_name = input_file.name.replace(".png", "_output.png")

            # Run the GEGL operation
            gegl_command = [
                gegl_path,
                input_file.name,
                "-o", output_file_name,
                '--', operation.replace('\n', ' ')
            ]

            result = subprocess.run(' '.join(gegl_command), check=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"GEGL command failed with error: {result.stderr.decode('utf-8')}")
            
            # wait for 500ms to ensure the file is written
            time.sleep(1.0)

            # Load the output image
            if include_alpha:
                output_pil_image = Image.open(output_file_name).convert("RGBA")
            else:
                output_pil_image = Image.open(output_file_name).convert("RGB")

        #os.remove(input_file.name)
        #os.remove(output_file_name)

        return (self.pil2tensor(output_pil_image), )
    
NODE_CLASS_MAPPINGS = {
    "GEGL": GEGL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GEGL": "GEGL"
}