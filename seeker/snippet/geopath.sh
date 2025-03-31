#date: 2025-03-31T17:13:06Z
#url: https://api.github.com/gists/13e00e5a3ceb9ca779be291170b53102
#owner: https://api.github.com/users/venam

#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <destination>"
    exit 1
fi

destination=$1
stdbuf -oL tracepath -n $destination | while read line; do
    ip=$(echo $line | grep -oP '\d+\.\d+\.\d+\.\d+')
    if [[ -n $ip ]]; then
        curl -s "https://ipinfo.io/$ip/json" | jq -r --arg line "$line" '"\($line) - City: \(.city), Country: \(.country)"'
    fi
done