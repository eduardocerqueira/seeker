#date: 2024-03-13T17:02:40Z
#url: https://api.github.com/gists/9cc2e3fd1199f07fef001d718b776eee
#owner: https://api.github.com/users/TunaCici

#!/usr/bin/env bash

if [ $# -eq 0 ]; then
        echo "Usage ./get-affinities.sh <PID>"
        exit 1
fi

pid=$1

if [ ! -d "/proc/${pid}" ]; then
        echo "Process with PID ${pid} does not exist."
        exit 1 
fi

name=$(cat /proc/${pid}/comm)
echo "[x] ${pid} (${name})"

for tid in $(ls /proc/${pid}/task/); do
        status="/proc/${pid}/task/${tid}/status"

        if [ -e "$status" ]; then
                name=$(cat /proc/${pid}/task/${tid}/comm)
                affinity=$(cat ${status} | grep -w "Cpus_allowed_list" | grep -oE '[0-9,-]+')

                echo "[x] \-- ${tid} ${name}, Core Affinity: ${affinity}"
        fi
done
