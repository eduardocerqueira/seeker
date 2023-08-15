#date: 2023-08-15T16:50:34Z
#url: https://api.github.com/gists/6dc0667cb739b5d7430a31774dad7499
#owner: https://api.github.com/users/sphillips-bdai

#!/bin/bash

mkdir -p movie
NUMFRAMES=$(ls *png | wc -l)
XMIN=$(identify *png | grep -o "[0-9]*x" | uniq | grep -o "[0-9]*" | sort -n | tail -1)
YMIN=$(identify *png | grep -o "x[0-9]*" | uniq | grep -o "[0-9]*" | sort -n | tail -1)
VIDSIZE=${XMIN}x${YMIN}
for x in $(ls *png)
do
    echo $x out of $NUMFRAMES
    convert $x -resize $VIDSIZE -background black -gravity center -extent $VIDSIZE movie/$x
done
cd movie
ffmpeg -framerate 10 -s $VIDSIZE -i '%04d.png' -vcodec libx264 -crf 25 demo.mp4