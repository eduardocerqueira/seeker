#date: 2021-11-02T17:10:49Z
#url: https://api.github.com/gists/601fd23acc612ea313d189b83d8a45c5
#owner: https://api.github.com/users/aadityarajkumawat

#!/bin/bash

command=$1

if [ $command = "start" ]; then
    # start backend
    cd ./backend
    yarn start &
    # go back to root
    cd ..
    # start frontend
    cd frontend
    yarn start &
elif [ $command = "stop" ]; then
    output=$(lsof -i tcp:4001)
    IFS=' ' read -ra ADDR <<< "$IN"
    for i in "${ADDR[@]}"; do
        # process "$i"
    done
    echo "$output"
else
    echo "please enter a valid command"
fi
