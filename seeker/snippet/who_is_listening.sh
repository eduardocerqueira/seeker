#date: 2021-09-02T17:09:27Z
#url: https://api.github.com/gists/594d426426d3c2b5975c7cc913513ada
#owner: https://api.github.com/users/curena

#! /bin/bash

# Usage: ./who_is_listening.sh <PORT>

echo "What process is listening to $1"
lsof -nP -i4TCP:$1 | grep LISTEN