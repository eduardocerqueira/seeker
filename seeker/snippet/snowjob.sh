#date: 2023-12-15T17:00:17Z
#url: https://api.github.com/gists/4252f260a80bb392cb23da586efc5f16
#owner: https://api.github.com/users/yowmamasita

#!/bin/bash

LINES=$(tput lines)
COLUMNS=$(tput cols)

declare -A snowflakes
declare -A lastflakes

clear

function move_flake() {
    i="$1"

    if [ "${snowflakes[$i]}" = "" ] || [ "${snowflakes[$i]}" = "$LINES" ]; then
        snowflakes[$i]=0
    else
        if [ "${lastflakes[$i]}" != "" ]; then
            printf "\033[%s;%sH \033[1;1H " ${lastflakes[$i]} $i
        fi
    fi

    printf "\033[%s;%sH❄\033[1;1H" ${snowflakes[$i]} $i

    lastflakes[$i]=${snowflakes[$i]}
    snowflakes[$i]=$((${snowflakes[$i]}+1))
}

while :
do
    i=$(($RANDOM % $COLUMNS))

    move_flake $i

    for x in "${!lastflakes[@]}"
    do
        move_flake "$x"
    done

    sleep 0.1
done
