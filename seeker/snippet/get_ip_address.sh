#date: 2024-04-04T16:47:34Z
#url: https://api.github.com/gists/4ab4bc7b4b19cb67acef954c9b941e75
#owner: https://api.github.com/users/vi7

#!/usr/bin/env bash

INTERFACE_NAME=eth0

ip -family inet -json addr show $INTERFACE_NAME | jq --raw-output .[0].addr_info[0].local
