#date: 2025-01-10T17:05:03Z
#url: https://api.github.com/gists/0530bfef2e773f5a71fc34884021352c
#owner: https://api.github.com/users/stefanocoretta

#!/bin/bash

if [[ $# -eq 0 ]]; then
    # If no folder provided, open the current folder in Positron
    positron .
elif [[ -d "$1" ]]; then
    # If a folder is provided, open that folder in Positron
    positron "$1"
fi