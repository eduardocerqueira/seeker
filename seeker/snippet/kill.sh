#date: 2024-09-30T16:55:20Z
#url: https://api.github.com/gists/7947d514c357ea7639579ae46fceaf49
#owner: https://api.github.com/users/Complexlity

#!/bin/bash
Process to kill port or consecutive ports on mac

kill_port() {
    local PORT=$1
    echo "Checking port $PORT"
    pids=$(lsof -t -i tcp:$PORT)
    if [ -z "$pids" ]; then
        echo "No processes found running on port $PORT."
    else
        echo "Killing processes on port $PORT with PIDs: $pids"
        for pid in $pids; do
            kill -9 $pid
            if [ $? -eq 0 ]; then
                echo "Successfully killed process $pid on port $PORT"
            else
                echo "Failed to kill process $pid on port $PORT"
            fi
        done
    fi
}

if [ $# -eq 0 ]; then
    echo "Wrong usage"
    echo "$0 <start_port> <end_port>"
    exit 1
fi

# Check if it's a range or individual ports
if [ $# -eq 2 ] && [ $2 -gt $1 ]; then
    # It's a range
    for PORT in $(seq $1 $2); do
        kill_port $PORT
    done
else
    # Individual ports
    for PORT in "$@"; do
        kill_port $PORT
    done
fi

echo "All processes have been removed."