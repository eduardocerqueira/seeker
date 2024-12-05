#date: 2024-12-05T17:06:03Z
#url: https://api.github.com/gists/181756472003152136f7bc0284a0dec1
#owner: https://api.github.com/users/Ne0n09

#!/bin/bash

# Run script with playlist name in the format of 'your playlist'
# If the playlist has spaces in the name put quotes around it

# Define the base directory
BASE_DIR="/home/fpp/media/playlists"

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: No playlist name provided. Please provide a playlist name."
    exit 1
fi

# Set the PLAYLIST variable from the $1 argument
PLAYLIST="$1"

# Remove any surrounding spaces
PLAYLIST=$(echo "$PLAYLIST" | xargs)

# Construct the full file path
INPUT_FILE="$BASE_DIR/$PLAYLIST.json"

# Check if the playlist file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found in directory '$BASE_DIR'."
    exit 1
fi

# Shuffle the JSON array using Python
python3 -c "
import json, random
with open('$INPUT_FILE', 'r') as f:
    data = json.load(f)
data['mainPlaylist'] = random.sample(data['mainPlaylist'], len(data['mainPlaylist']))
with open('$INPUT_FILE', 'w') as f:
    json.dump(data, f, indent=4)
"

echo "Randomized sequences written back to $INPUT_FILE"