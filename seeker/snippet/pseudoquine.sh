#date: 2023-04-12T17:00:57Z
#url: https://api.github.com/gists/1de625beadfe49e1e073feaee4e6c870
#owner: https://api.github.com/users/NWuensche

#!/bin/sh
#Program that almost writes itself into itself (but not quite), demonstrate that bash is executed line by line rather than whole file

CURR_FILENAME=$0
echo "Hello World"
sed -n '5,6 p' $CURR_FILENAME >> $CURR_FILENAME #Write 4th and 5th line into same file (i.e. echo + sed)