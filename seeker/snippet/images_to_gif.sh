#date: 2024-03-20T16:59:21Z
#url: https://api.github.com/gists/8870f1917ba58a02cab40a21fe316bb1
#owner: https://api.github.com/users/clairefreehafer

#!/bin/bash

input=$1
# TODO: figure out ideal framerate(s)
framerate=25
output=$2
boomerang=$3

if [[ ${boomerang} ]]; then
  filter="[0]trim=start_frame=1:end_frame=9,setpts=PTS-STARTPTS,reverse[r]; [0][r]concat,split[a][b]; [a]palettegen[p]; [b][p]paletteuse"
else
  filter="[0:v]split[a][b]; [a]palettegen[p]; [b][p]paletteuse"
fi

ffmpeg -framerate ${framerate} -start_number 0 -start_number_range 9 -i ${input} -filter_complex "${filter}" ${output}
