#date: 2025-07-23T17:18:04Z
#url: https://api.github.com/gists/eb60bbb46cd50577402003240a70cc25
#owner: https://api.github.com/users/TheCrazyGM

#!/bin/bash

find_pids() {
    local port=$1

    if command -v ss &>/dev/null; then
        ss -lptn "sport = :$port" 2>/dev/null \
        | grep -oP 'pid=\K[0-9]+' \
        | sort -u           # one PID per line
    elif command -v lsof &>/dev/null; then
        lsof -t -i:"$port"
    else
        echo "Neither ss nor lsof found." >&2
        return 1
    fi
}

#— main —#
[ -z "$1" ] && { echo "Usage: $0 <port>"; exit 1; }
PORT=$1

# Get PIDs as a space-separated list
PIDS=$(find_pids "$PORT" | tr '\n' ' ')

if [ -n "$PIDS" ]; then
    echo "Killing processes on port $PORT: $PIDS"
    # shellcheck disable=SC2086  # intentional word-splitting so each PID is separate
    kill -9 $PIDS
    echo "Process(es) killed."
else
    echo "No process found on port $PORT"
fi
