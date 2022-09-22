#date: 2022-09-22T17:13:06Z
#url: https://api.github.com/gists/4adf9d249b54812fea0a0c1329553ace
#owner: https://api.github.com/users/LiliwoL

# requirement! install imagemagick 
# brew install imagemagick
# sudo port install imagemagick
# or build from source here http://www.imagemagick.org/script/binary-releases.php

# take every png in the folder and smash into a gif with a frame rate of 0.5 sec
convert -delay 50 /path/ti/my/folder/*.png my.gif 