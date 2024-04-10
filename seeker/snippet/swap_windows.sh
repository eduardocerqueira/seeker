#date: 2024-04-10T16:55:24Z
#url: https://api.github.com/gists/13c99f8fb339d44eda8f8fd685c159d6
#owner: https://api.github.com/users/Eruvae

#!/bin/bash

while read -ra line; do
    if [ "${line[0]}" == "*" ]; then
        IFS='x' read -ra res <<< "${line[1]}"
        screen_width="${res[0]}"
        break
    fi
done < <(wmctrl -d | tr -s ' ' | cut -d ' ' -f 2,4)

half_screen_width=$((screen_width/2))

while read -ra line; do
    window_id="${line[0]}"
    x=$(xwininfo -id $window_id | grep "Absolute upper-left X" | awk '{print $NF}')
    if [ "$x" -lt "$half_screen_width" ]; then
        new_x=$((x+half_screen_width))
    else
        new_x=$((x-half_screen_width))
    fi

    wmctrl -ir $window_id -e 0,$new_x,-1,-1,-1

done < <(wmctrl -l | tr -s ' ' | cut -d ' ' -f 1)