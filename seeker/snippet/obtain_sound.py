#date: 2025-01-20T16:33:40Z
#url: https://api.github.com/gists/bb90bd6771b14e8b9575ad2d678bb111
#owner: https://api.github.com/users/skpub

import json
import os
import shutil

sound_registry_file = "assets/indexes/17.json"
objects_dir = "assets/objects"
dst_dir = "output"

def my_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"Failed to copy {src} to {dst}: {e}")

with open(sound_registry_file, "r") as f:
    sound_registry = json.load(f)["objects"]

for key, value in sound_registry.items():
    if key.endswith(".ogg"):
        sound_filename = value["hash"]
        subdir = sound_filename[:2]
        print(f"{objects_dir}/{subdir}/{sound_filename}", "  to  ", f"{dst_dir}/{key}")
        my_copy(f"{objects_dir}/{subdir}/{sound_filename}", f"{dst_dir}/{key}")
