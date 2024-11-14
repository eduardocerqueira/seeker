#date: 2024-11-14T16:52:40Z
#url: https://api.github.com/gists/68ee794b34880ea5760ab3c9ad3a9cf6
#owner: https://api.github.com/users/hiranp

#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/path/to/source"
DEST_DIR="/path/to/destination"
START_DATE="2024-10-01"               # Change this to the start date of the range
END_DATE="2024-10-15"                 # Change this to the end date of the range
EXTENSIONS=("*.txt" "*.log" "*.conf") # Add the specific extensions you want to match

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Find and copy files with specific extensions and within the date range from SOURCE_DIR to DEST_DIR
for EXT in "${EXTENSIONS[@]}"; do
  find "$SOURCE_DIR" -type f -name "$EXT" -newermt "$START_DATE" ! -newermt "$END_DATE +1 day" -exec cp --parents {} "$DEST_DIR" \;
done

echo "Files with specific extensions and within the date range have been copied from $SOURCE_DIR to $DEST_DIR."
