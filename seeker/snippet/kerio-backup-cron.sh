#date: 2021-10-20T17:02:07Z
#url: https://api.github.com/gists/f252cec4357c14f8a8860eb3bee02623
#owner: https://api.github.com/users/orinocoz

#!/bin/sh -e

LOGDIR="/var/log/${0##*/}"
TSTAMP="$(date +%Y-%m-%d-%H%M)"

mkdir -p "$LOGDIR"

VERBOSE=1 /root/kerio-backup/kerio-backups-to-hetzner > "$LOGDIR/$TSTAMP.log" 2>&1

find "$LOGDIR/" -name "*.log" -type f -mtime +7 -delete