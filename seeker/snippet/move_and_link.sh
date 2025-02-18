#date: 2025-02-18T16:43:38Z
#url: https://api.github.com/gists/2d8c522776b9ee7dd8ec413e9992d754
#owner: https://api.github.com/users/hexiaoxiao-cs

#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_folder> <destination_parent_folder>"
    exit 1
fi

SOURCE_FOLDER="$1"
DESTINATION_PARENT="$2"
FOLDER_NAME="$(basename "$SOURCE_FOLDER")"
DESTINATION_FOLDER="$DESTINATION_PARENT/$FOLDER_NAME"

# Ensure the source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder does not exist."
    exit 1
fi

# Ensure the destination parent folder exists
if [ ! -d "$DESTINATION_PARENT" ]; then
    echo "Error: Destination parent folder does not exist."
    exit 1
fi

# Move the folder using rsync
rsync -av --remove-source-files "$SOURCE_FOLDER/" "$DESTINATION_FOLDER/"
rm -r "$SOURCE_FOLDER"

# Create a symbolic link from the original location to the new location
ln -s "$DESTINATION_FOLDER" 

echo "Successfully moved '$SOURCE_FOLDER' to '$DESTINATION_FOLDER' and created a symbolic link."
