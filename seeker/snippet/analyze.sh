#date: 2023-11-06T16:58:56Z
#url: https://api.github.com/gists/ebb2a8b2435da1c8f2e71f4300ce3d72
#owner: https://api.github.com/users/chriskrogh

#!/bin/bash

# Initialize a variable to hold the total size
total_size=0
#paths_file="treatment.txt"
paths_file="control.txt"

# Check if the paths file exists
if [ -e $paths_file ]; then
    # Loop through each line in paths
    while IFS= read -r line; do
        # Check if the file exists
        if [ -e "$line" ]; then
            # Use gzip and wc to get the size of the compressed file and add it to the total size
            file_size=$(gzip -c "$line" | wc -c)
            total_size=$((total_size + file_size))
        else
            echo "File not found: $line"
        fi
    done <$paths_file

    # Function to format size in human-readable format
    format_size() {
        local size="$1"
        if ((size < 1024)); then
            echo "${size}B"
        elif ((size < 1048576)); then
            echo "$((size / 1024))KB"
        elif ((size < 1073741824)); then
            echo "$((size / 1048576))MB"
        else
            echo "$((size / 1073741824))GB"
        fi
    }

    # Print the total size in human-readable format
    formatted_size=$(format_size "$total_size")
    echo "Total size of compressed files: $formatted_size"
else
    echo "paths file not found."
fi
