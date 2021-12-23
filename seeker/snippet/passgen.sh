#date: 2021-12-23T16:56:18Z
#url: https://api.github.com/gists/6e887b4e273895af4d917b3e214efb1e
#owner: https://api.github.com/users/mikhdm

#!/bin/bash

if [ -z "$1" ]; then
	PATTERN="[A-Za-z0-9_]"
else
	PATTERN=$1
fi
if [ -z "$2" ]; then
	N=16
else
	N=$2
fi

echo $(cat /dev/urandom | LC_ALL=C tr -cd "$PATTERN" | head -c $N)