#date: 2024-03-19T17:07:07Z
#url: https://api.github.com/gists/8ebb4c32be7391e61a7ee098e9fc331f
#owner: https://api.github.com/users/kgriffs

#!/bin/bash

# Set the directory where your MP3 files are located
input_dir="."

# Set the desired output bitrate (in bits per second)
output_bitrate="192k"  # Adjust this value as needed

# Loop through each MP3 file in the directory
for file in "$input_dir"/*.mp3; do
    # Extract the filename without extension
    filename=$(basename -- "$file")
    filename_no_ext="${filename%.*}"

    # Output filename for the extended AAC file
    output_file="${input_dir}/${filename_no_ext}_1_hour.aac"

    # Use FFmpeg to extend the MP3 file to one hour with specified output quality and preserve metadata
    ffmpeg -i "$file" -af "aloop=loop=3600" -t 3600 -c:a aac -b:a "$output_bitrate" "$output_file"
done
