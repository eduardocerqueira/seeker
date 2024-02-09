#date: 2024-02-09T16:46:48Z
#url: https://api.github.com/gists/11ebcadd5c17644071e6afde85c8b8b3
#owner: https://api.github.com/users/chrisliebaer

#!/bin/bash

# sample:
#
# ./avatar.sh YOU_BOT_TOKEN avatar.gif

set -e

if [ $# -ne 2 ]; then
	echo "Usage: "**********"
	exit 1
fi

token= "**********"
file="$2"

touch payload.json
echo "{\"avatar\": \"data:image/gif;base64," > payload.json
cat $file | base64 -w 0 >> payload.json
echo "\"}" >> payload.json
curl -X PATCH -H "Content-Type: "**********": Bot $token" "https://discord.com/api/users/@me" -d @payload.json
rm payload.json
son
rm payload.json
