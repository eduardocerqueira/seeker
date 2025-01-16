#date: 2025-01-16T16:37:30Z
#url: https://api.github.com/gists/c1099058aef0a0b9ea06f49e2a5da5c0
#owner: https://api.github.com/users/dogue

#!/usr/bin/env bash

if [ -z "$1" ]; then
    xrandr -q --verbose | grep "Brightness: " | awk 'NR==1 {print $2}'
    exit 0
fi

brightness=$1
displays=($(xrandr -q | grep " connected" | awk '{print $1}'))
cmd="xrandr"
for ((i=0; i<${#displays[@]}; i++)); do
    cmd="$cmd --output ${displays[$i]} --brightness $brightness"
done

$cmd