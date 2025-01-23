#date: 2025-01-23T16:59:06Z
#url: https://api.github.com/gists/9469d8f859f49378dbba9e5d461b9b62
#owner: https://api.github.com/users/Cemu0

#!/bin/bash

while true; do
  # Capture a frame from the RTSP stream
  filename="$(date +\%Y-\%m-\%d_\%H-\%M-\%S).png"
  filepath="/output2/$(date +\%Y-\%m-\%d_\%H-\%M-\%S).png"
  ffmpeg -rtsp_transport tcp -i 'rtsp://admin:L2B924E3@192.168.2.42:554/cam/realmonitor?channel=1&subtype=0' -vframes 1 -q:v 1 -s hd1080 -pix_fmt yuv420p "$filepath"
  
  # Append the filename to files.txt
  echo "$filename" >> /output2/files.txt
  
  # Sleep for 60 seconds
  sleep 60
done