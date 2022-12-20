#date: 2022-12-20T17:00:56Z
#url: https://api.github.com/gists/89ec933bdf64b539ae20235fcdb407b1
#owner: https://api.github.com/users/roger-hope

import os
import shutil

# A Pyhon script that copies all files in a folder to their individual folders. Rename each file to "page.xml" and increment each folder starting with zero.


# Set the source (/path/to/source/folder) and destination (/path/to/destination/folder) directories
source_dir = 'page'
dest_dir = 'pagesoutput/pages'

# Get a list of all files in the source directory
files = os.listdir(source_dir)

# Iterate over the files
for i, file in enumerate(files):
    # Construct the full path to the file
    source_path = os.path.join(source_dir, file)

    # Check if the file is a regular file (not a directory)
    if os.path.isfile(source_path):
        # Create the destination directory
        dest_subdir = os.path.join(dest_dir, "page" + str(i))
        os.makedirs(dest_subdir, exist_ok=True)

        # Construct the full path to the destination file
        dest_path = os.path.join(dest_subdir, 'page'  + str(i) + '.xml')

        # Copy the file to the destination
        shutil.copy(source_path, dest_path)
