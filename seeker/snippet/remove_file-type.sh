#date: 2023-07-18T17:05:52Z
#url: https://api.github.com/gists/251b54396bceb24d21e7f916a00104a2
#owner: https://api.github.com/users/SOSANA

#!/bin/bash
# remove all files named "file-type" in the root directory and subfolders
# bash remove_files.sh

# Get the current directory
current_dir=$(pwd)

# Initialize counter for file-type files
file_type_count=0

# Find and delete files named "file-type"
while IFS= read -r -d '' file; do
  rm -f "$file"
  echo "Deleted: $file"
  ((file_type_count++))
done < <(find "$current_dir" -type f -name "file-type" -print0)

# Log the completion message with the count
echo "Deletion process completed."
echo "\"file-type\" files deleted: $file_type_count"
