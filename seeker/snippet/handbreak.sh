#date: 2023-10-02T16:57:06Z
#url: https://api.github.com/gists/6eab09def170690bcab8da5a8db34b5c
#owner: https://api.github.com/users/ademar111190

#!/usr/bin/env bash
filename=$(basename -- "$1")
filename="${filename%.*}"
ffmpeg -i $1 -vcodec libx265 -crf 30 ${filename}-min.mp4
