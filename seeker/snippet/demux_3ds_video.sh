#date: 2021-12-08T17:04:30Z
#url: https://api.github.com/gists/b2db5488b377478ef2fa318749f137dd
#owner: https://api.github.com/users/ivkowalenko

#!/usr/bin/env bash

# Demuxes 3D videos from Nintendo 3DS into two (left and right) videos. 

# REQUIRES FFMPEG, can be installed with Homebrew on Mac
# Install FFmpeg on ubuntu with: http://askubuntu.com/a/451158
# Forked from https://gist.github.com/scascketta/9d096d1ccea575b0f375 to include fixes for correct ffmpeg mapping of video channels and processing of audio.

if [ $# -ne 1 ]
then
    echo "Usage: ./demux_3ds_video.sh <video>"
    exit 1
fi

# check if ffmpeg is installed
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required and not installed.  Aborting." >&2; exit 1;}

VIDEO_FNAME=$1
VIDEO_NAME=${VIDEO_FNAME:0:${#VIDEO_FNAME}-4}
VIDEO_LEFT=$VIDEO_NAME"_left"
VIDEO_RIGHT=$VIDEO_NAME"_right"
VIDEO_LEFT_FNAME=$VIDEO_LEFT".mov"
VIDEO_RIGHT_FNAME=$VIDEO_RIGHT".mov"

if [ ! -f $VIDEO_FNAME ]; then
    echo "$VIDEO_FNAME not found. Exiting."
    exit 1
fi

echo "Demux $VIDEO_NAME into $VIDEO_LEFT_FNAME and $VIDEO_RIGHT_FNAME..."

ffmpeg -i $VIDEO_FNAME -c:v copy -c:a pcm_s16le -map 0:v:0 -map 0:a:0 $VIDEO_LEFT_FNAME
if [ $? -eq 0 ]; then
    echo "Demuxed $VIDEO_FNAME into $VIDEO_LEFT_FNAME."
else
    echo -e "Error occurred while demuxing $VIDEO_FNAME into $VIDEO_LEFT_FNAME."
    exit 1
fi

ffmpeg -i $VIDEO_FNAME -c:v copy -c:a pcm_s16le -map 0:v:1 -map 0:a:0 $VIDEO_RIGHT_FNAME
if [ $? -eq 0 ]; then
    echo "Demuxed $VIDEO_FNAME into $VIDEO_RIGHT_FNAME."
else
    echo -e "Error occurred while demuxing $VIDEO_FNAME into $VIDEO_RIGHT_FNAME."
    exit 1
fi