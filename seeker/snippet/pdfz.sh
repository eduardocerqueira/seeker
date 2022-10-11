#date: 2022-10-11T17:26:01Z
#url: https://api.github.com/gists/15ef6e5644be51f4c0765318649571e2
#owner: https://api.github.com/users/oletizi

#!/usr/bin/env zsh

filename=$1

gs \
    -sDEVICE=pdfwrite \
    -dCompatibilityLeve=1.4 \
    -dPDFSETTINGS=/ebook \
    -dNOPAUSE \
    -dQUIET \
    -dBATCH \
    -sOutputFile=${filename:r}.z.pdf $filename