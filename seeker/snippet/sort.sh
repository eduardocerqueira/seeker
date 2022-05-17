#date: 2022-05-17T17:06:21Z
#url: https://api.github.com/gists/f3e32952ab5c1577995039710b1c2d40
#owner: https://api.github.com/users/blackhalt

#!/bin/bash

dir_size=5000
dir_name="0"
n=$((`find . -maxdepth 1 -type f | wc -l`/$dir_size+1))
for i in `seq 1 $n`;
do
    mkdir -p "$dir_name$i";
    find . -maxdepth 1 -type f | head -n $dir_size | xargs -i mv "{}" "$dir_name$i"
done
