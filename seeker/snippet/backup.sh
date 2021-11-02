#date: 2021-11-02T17:02:00Z
#url: https://api.github.com/gists/87ca68ad79b7dc5800cd810056c25721
#owner: https://api.github.com/users/jogerj

#!/bin/bash

MCRCON_PATH="/usr/local/bin"
BACKUP_PATH="/home/opc/mc_backup"
MC_PATH="/home/opc/minecraft"

IP="127.0.0.1"
PORT="25575"
PASS="[password]"

function rcon {
    $MCRCON_PATH/mcrcon -H $IP -P $PORT -p $PASS "$1"
}

rcon "save-off"
rcon "save-all"
tar -cvpzf $BACKUP_PATH/server-$(date +%F_%R).tar.gz $MC_PATH
rcon "save-on"
## Delete older backups
find $BACKUP_PATH -type f -mtime +7 -name '*.gz' -delete

##file paths and other details omitted for this public copy, just replace them with your information if you want to use this
##i have this set up on crontab to run every 6 hours
##enjoy :)