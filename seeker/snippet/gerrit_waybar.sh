#date: 2023-05-29T17:09:41Z
#url: https://api.github.com/gists/42915dbbe71583e2d421bc5e6fe6fccd
#owner: https://api.github.com/users/dgulino

#!/bin/bash

# config.jsonc example:
#    "custom/gerrit": {
#        "format": "G{}",
#        "exec": "/home/$USER/.config/waybar/gerrit_waybar.sh",
#        "interval": 60,
#        "return-type": "json"
#   },

gerrit_user='$USERNAME'
gerrit_host='$GERRIT_HOST'
gerrit_port='29418'

# hack to find user's ssh-agent
export SSH_AUTH_SOCK="$(find /tmp/ -type s -path '/tmp/ssh-*/agent.*' -user $(whoami) 2>/dev/null | tail -1)"
gerrit_response=`ssh -o ConnectTimeout=3 $gerrit_user@$gerrit_host -p $gerrit_port gerrit query --format=JSON --all-approvals status:open reviewer:$gerrit_user 2>/dev/null` 
ret=$?
if [[ $ret -ne 0 ]]; then
    echo "{\"text\":\"x\", \
\"tooltip\": \"href=https://$gerrit_host/#/dashboard/self\" \
}"
    exit 0 
fi
row_count=`echo $gerrit_response | jq '.rowCount'`
if [[ $row_count -lt 1 ]]; then
    echo "{\"text\":$row_count, \
\"tooltip\": \"href=https://$gerrit_host/#/dashboard/self\" \
}"
else
    echo "{\"text\":\"⚡$row_count\", \
\"tooltip\": \"href=https://$gerrit_host/#/dashboard/self\" \
}"
fi