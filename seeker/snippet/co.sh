#date: 2024-04-10T17:09:26Z
#url: https://api.github.com/gists/396b217c6b4e4c6483c4d6902d416f63
#owner: https://api.github.com/users/stefanocoretta

#!/bin/bash

if [[ $# -eq 0 ]]; then
    # If no folder provided, open the current folder in VS code
    code .
elif [[ -d "$1" ]]; then
    # If a folder is provided, open that folder in VS code
    code "$1"
fi