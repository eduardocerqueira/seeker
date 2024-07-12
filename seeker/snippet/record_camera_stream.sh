#date: 2024-07-12T17:11:47Z
#url: https://api.github.com/gists/91fb40ef648bdc173cce793145b7deb6
#owner: https://api.github.com/users/lebedev-a

#!/bin/bash
# Bash Unofficial strict mode
set -euo pipefail

CAMERA_HOST="$1"

#find out where the script is located iself
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null; pwd)"
#read password from file, the same for all the cameras
PASS=$( < "${SCRIPT_DIR}"/pwd)

#2024-07-20
TODAY=$(date +'%Y-%m-%d')

TARGET_DIR=/opt/cameras/"${CAMERA_HOST}"

#ensure directory exists
mkdir -p "${TARGET_DIR}"/"${TODAY}"

#Save stream into todays folder
ffmpeg -i "rtsp://admin:${PASS}@${CAMERA_HOST}:8554/1080p?video=all&audio=all" \
    -err_detect aggressive  -c copy -f segment -segment_time 59  \
    -reset_timestamps  1  -strftime 1  \
    -segment_format mp4  "${TARGET_DIR}"/"%Y-%m-%d/%Y-%m-%d-%H-%M-%S.mp4"
