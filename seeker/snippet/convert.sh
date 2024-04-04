#date: 2024-04-04T16:55:20Z
#url: https://api.github.com/gists/dde9705eebdb60285a06994856a3e604
#owner: https://api.github.com/users/1pxone

#!/bin/bash

# Iterate over each .mp4 file in the current directory
for file in *.mp4; do
  # Extract the base name without the extension
  base_name=$(basename "$file" .mp4)

  # Create a directory with the same name as the file
  mkdir -p "$base_name"

  # Run ffmpeg with the specified configuration
  ffmpeg -i "$file" -codec:v libx264 -b:v 800k -preset veryslow -codec:a aac -b:a 128k -start_number 0 -hls_time 10 -hls_list_size 0 -f hls "$base_name/video.m3u8"
done

