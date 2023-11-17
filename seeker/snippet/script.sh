#date: 2023-11-17T17:09:34Z
#url: https://api.github.com/gists/b3656565c1c3d68c0996a14b94b74e3d
#owner: https://api.github.com/users/auryn31

#!/bin/bash

# Input file containing timestamps and text
input_file="input.txt"
output_file="output.txt"
# Original video file
original_file="meme.mp4"
# Output directory for the new files
output_directory="output"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"


> "$output_file"
# Read the file line by line
while IFS= read -r c_line ; do
  if [ -n "$l_line" ]; then
    timestamp1=$(echo "$l_line" | awk '{print $1}')
    timestamp2=$(echo "$c_line" | awk '{print $1}')
    input_time=$timestamp2

    # Split the input time string at the colon
    IFS=':' read -ra time_parts <<< "$input_time"

    # Extract minutes and seconds
    minutes=${time_parts[0]}
    seconds=${time_parts[1]}

    # Convert input time to seconds
    total_seconds=$((minutes * 60 + 10#$seconds))

    # Add 1 second
    total_seconds=$((total_seconds + 1))

    # Calculate new minutes and seconds
    new_minutes=$((total_seconds / 60))
    new_seconds=$((total_seconds % 60))
    timestamp2=$(printf "%02d:%02d" "$new_minutes" "$new_seconds")
    title=$(echo "$l_line" |  cut -d' ' -f2-)
    echo "line 1: $c_line"

    echo "Run $timestamp1 $timestamp2 $title"
    echo "00:$timestamp1 00:$timestamp2 $title" >> "$output_file"
    ffmpeg -nostdin -loglevel 0 -i "$original_file" -ss "$timestamp1" -to "$timestamp2" "$output_directory/$title.mp4" 
    echo "done $title"
  fi
  l_line=$c_line
  # Extract the first word of the current line
done < "$input_file"

echo "done"