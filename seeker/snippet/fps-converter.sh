#date: 2024-01-02T16:54:18Z
#url: https://api.github.com/gists/ea61f310c5da9bea0e4d916dbaf0da7e
#owner: https://api.github.com/users/lfbittencourt

#!/bin/bash

for file in *.MP4; do
  echo "Processing $file..."
  output="${file%.MP4}-60fps.mp4"
  ffmpeg -i "$file" -filter:v fps=60 "$output"
done
