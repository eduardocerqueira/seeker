#date: 2022-06-14T16:58:32Z
#url: https://api.github.com/gists/ca4289679b897ad9cadad556a3504afc
#owner: https://api.github.com/users/adityajadhav99

#! /usr/bin/bash

COND=${1?Error: input not provided}

if [ $COND == '0' ]
then
	echo "running roscore"
    roscore
elif [ $COND == '1' ]; then
	echo "Input read as 1" 
elif [ $COND == '--help' ]; then
	printf "Usage: ./`basename $0` 0 to run experiment 1\n./`basename $0` 1 to run experiment 2\n"
    exit 0
else
	echo "Enter either 0 or 1"
    exit 0
fi