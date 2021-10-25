#date: 2021-10-25T16:54:18Z
#url: https://api.github.com/gists/4e097caba41e32be26198b98edb5766c
#owner: https://api.github.com/users/brianjo

#!/bin/bash

gource \
    -s .03 \
    -1280x720 \
    --auto-skip-seconds .1 \
    --multi-sampling \
    --stop-at-end \
    --key \
    --highlight-users \
    --date-format "%d/%m/%y" \
    --hide mouse,filenames \
    --file-idle-time 0 \
    --max-files 0  \
    --background-colour 000000 \
    --font-size 25 \
    --output-ppm-stream - \
    --output-framerate 30 \
    | ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i - -b 65536K movie.mp4