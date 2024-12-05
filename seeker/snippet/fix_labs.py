#date: 2024-12-05T17:01:10Z
#url: https://api.github.com/gists/b1f21a303bfc5bdaec8f869388a9fda3
#owner: https://api.github.com/users/Nikh1l

import os
import shutil

root_dir = "D:\Productive\Plugins\SplitFire\Spitfire Audio - LABS"
source_folders = ["Patches", "Presets", "Samples"]

def reorganise_labs_folder():
    matching_folders = set()

    for folder in source_folders:
        folder_path = os.path.join(root_dir, folder)

        if os.path.exists(folder_path):
            sub_folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            if not matching_folders:
                matching_folders.update(sub_folders)
            else:
                matching_folders.intersection_update(sub_folders)
        
    for match in matching_folders:
        destination = os.path.join(root_dir, match)
        os.makedirs(destination, exist_ok=True)

        for folder in source_folders:
            source_path = os.path.join(root_dir, folder, match)
            dest_path = os.path.join(destination, folder)

            if os.path.exists(source_path):
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                print(f"Copied from {source_path} to {dest_path}")


if __name__ == "__main__":
    reorganise_labs_folder()