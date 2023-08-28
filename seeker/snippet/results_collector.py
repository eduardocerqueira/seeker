#date: 2023-08-28T16:54:34Z
#url: https://api.github.com/gists/0372a26f38dfe10c52f43fe5cdfc7c84
#owner: https://api.github.com/users/Taremeh

import os
import shutil
import argparse

def main(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Traverse the source folder and its subdirectories
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file == "trial.json":
                # Get the relative path from the source folder
                relative_path = os.path.relpath(root, source_folder)
                
                # Rename the file with the folder location
                new_file_name = f"{relative_path.replace(os.sep, '-')}-{file}"
                
                # Copy the trial.json file to the destination folder with the new name
                shutil.copy2(os.path.join(root, file), os.path.join(destination_folder, new_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename trial.json files from a source folder to a destination folder.")
    parser.add_argument("source_folder", help="Path to the source folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    args = parser.parse_args()
    
    main(args.source_folder, args.destination_folder)
