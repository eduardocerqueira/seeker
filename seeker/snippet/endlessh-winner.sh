#date: 2024-02-22T16:54:09Z
#url: https://api.github.com/gists/619d26afafd85484e23d7e6594028efe
#owner: https://api.github.com/users/Opa-

#!/bin/bash

TOP=$(journalctl -u endlessh.service | grep "time=" | awk '{split($11, time, "="); split($8, ip, ":"); print ip[4], time[2]/60}' | awk '{arr[$1]+=$2} END {for (i in arr) {print i,arr[i]}}' | sort -nr -k2 | head -n 30)

TABLE=""
while IFS= read -r WINNER; do
    IP=$(echo $WINNER | cut -d' ' -f1)
    TIME=$(echo $WINNER | cut -d' ' -f2)
    LOC=$(geoiplookup $IP | cut -d':' -f2 | tr "\n" " ")
    TABLE="$TABLE$IP;$LOC;$TIME min\n"
done <<< "$TOP"

echo -e $TABLE | column --table --separator ';'