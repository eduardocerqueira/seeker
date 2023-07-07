#date: 2023-07-07T16:54:23Z
#url: https://api.github.com/gists/fdd432b03c23624e4e9692eb29acb681
#owner: https://api.github.com/users/aginanjar

#!/bin/bash

ffmpeg -f x11grab -video_size 1920x1080 -framerate 25 -i $DISPLAY -c:v ffvhuff ~/Videos/ok.mkv