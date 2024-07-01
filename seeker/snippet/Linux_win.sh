#date: 2024-07-01T16:34:58Z
#url: https://api.github.com/gists/21c52aefec2689a46423e0901749c08d
#owner: https://api.github.com/users/jagannath-sahoo

#!/bin/bash

# Get the drive letter and Linux path arguments
drive_letter="$1"
linux_path="$2"

# Check if arguments are provided
if [ -z "$drive_letter" ] || [ -z "$linux_path" ]
then
  echo "Error: Please provide both drive letter and Linux file path as arguments."
  exit 1
fi

# Convert forward slashes (/) to backslashes (\) using sed
windows_path=$(echo "$linux_path" | sed 's/\//\\/g')

# Prepend drive letter with colon (:) and combine with path
windows_path="$drive_letter:$windows_path"

# Convert drive letter to lowercase (optional)
windows_path=$(tr '[:upper:]' '[:lower:]' <<< "$windows_path")

# Print the converted Windows path
echo "Windows path: $windows_path"
