#date: 2022-09-14T17:14:14Z
#url: https://api.github.com/gists/3d88fcdaedd4c7b945c7a8886fe754ea
#owner: https://api.github.com/users/jrwarwick

#!/usr/bin/bash
#https://askubuntu.com/a/686944/234023

if [ -z "$XDG_RUNTIME_DIR" ]; then
        XDG_RUNTIME_DIR=/run/user/$(id -u)
        if [ -d "$XDG_RUNTIME_DIR" ] && [ -w "$XDG_RUNTIME_DIR" ]; then
                export XDG_RUNTIME_DIR
        else
                unset XDG_RUNTIME_DIR
        fi
fi
CONTAINER_NAME=name_of_your_ms_logcollector_container

# General network connectvity check and report...
ping -c 4 1.1.1.1 || ( spd-say --sound-icon prompt ; spd-say --wait --ssml "<speak><prosody rate="x-slow">Attention: $HOSTNAME network connectivity degraded.</prosody> Please check network interfaces and reissue net-plan apply. <say-as interpret-as=\"spell-out\">$(ip -4 -h -brief link show eno1 | sed 's/\s\+/ is /g' | cut -f 1,2,3 -d' ')</say-as></speak>" )

# Now the log collector (process in container) status itself
STATUS_REPORT="$(docker exec -it $CONTAINER_NAME collector_status -p | egrep -i 'status:|last connect' |  sed -r 's/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g' | sed 's/:[0-9]\{2\}//' )"
echo "$STATUS_REPORT" | grep -i "status: ok"
if [ $? -eq 0 ] ; then
                echo nominal
        else
                spd-say --sound-icon prompt
                echo "Attention! $HOSTNAME Log Collector $STATUS_REPORT" | spd-say --pipe-mode --wait
fi
