#date: 2024-03-07T18:19:45Z
#url: https://api.github.com/gists/9e66f41ea053a88fee1691d83f73008f
#owner: https://api.github.com/users/jackbdu

#!/bin/bash

# Path to the file containing the list of filenames and URLs
urls_file="filenames_and_p5js_sketch_urls.txt"

# Check if the file exists
if [ ! -f "$urls_file" ]; then
    echo "Error: File '$urls_file' not found."
    exit 1
fi

# Loop through each line in the file
while IFS=, read -r filename url; do
    # Extract the sketch ID from the URL
    sketch_id=$(echo "$url" | sed 's#.*/\([^/]*\)$#\1#')

    # Construct the download URL
    download_url="https://editor.p5js.org/editor/projects/$sketch_id/zip"

    # Download the file using curl with the custom filename
    curl -o "$filename.zip" "$download_url"
done < "$urls_file"