#date: 2023-02-27T17:09:33Z
#url: https://api.github.com/gists/7f71754a7d0b8fc62d3d7d884f8b765c
#owner: https://api.github.com/users/gnurock

#!/bin/bash

# create png_files directory if it doesn't exist
if [ ! -d "png_files" ]; then
    mkdir png_files
fi

# move all .png files to png_files directory
find . -maxdepth 1 -type f -name "*.png" -print0 | pv -0 | xargs -0 -I {} mv {} png_files/
