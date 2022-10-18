#date: 2022-10-18T17:21:15Z
#url: https://api.github.com/gists/d3b1de314d0d351f72bf509441a23246
#owner: https://api.github.com/users/edgeloom

#!/bin/bash

function enable_core() {
    bash -c "echo 1 > \"/sys/devices/system/cpu/cpu$1/online\""
    echo "Set Core $1 : enabled"
}

function disable_core() {
    bash -c "echo 0 > \"/sys/devices/system/cpu/cpu$1/online\""
    echo "Set Core $1 : disabled"
}

function enable_game_mode() {
    if [ "$UID" != "0" ]; then
        echo "Run this script as root"
    else
        numberOfCores=$(nproc --all)
        for ((x=0; x < $numberOfCores; x++)); do
            if [ $(( $x % 2 )) -eq 0 ]; then
                enable_core $x
            else
                disable_core $x
            fi
        done
    fi
}

function disable_game_mode() {
    if [ "$UID" != "0" ]; then
        echo "Run this script as root"
    else
        numberOfCores=$(nproc --all)
        for ((x=0; x < $numberOfCores; x++)); do
            enable_core $x
        done
    fi
}

if [ "$1" == "enable" ]; then
    enable_game_mode
else
    disable_game_mode
fi
