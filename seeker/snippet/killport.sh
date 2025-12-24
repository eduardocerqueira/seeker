#date: 2025-12-24T17:02:42Z
#url: https://api.github.com/gists/8f887eccee15205d6faf8d3acaffceae
#owner: https://api.github.com/users/thisguymartin

#!/bin/bash
# killport: Kill process on a given port

PORT=$1

if [ -z "$PORT" ]; then
    echo "Usage: killport <port>"
    exit 1
fi

PID=$(lsof -ti :$PORT)

if [ -z "$PID" ]; then
    echo "Nothing running on port $PORT"
else
    echo "Killing PID $PID on port $PORT"
    kill -9 $PID
    echo "✅ Done"
fi
