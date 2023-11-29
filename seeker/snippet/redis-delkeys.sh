#date: 2023-11-29T16:46:50Z
#url: https://api.github.com/gists/495e8115611e1ac5a32584ba66dc8a15
#owner: https://api.github.com/users/zeroxs

#!/bin/sh
#
# Usage: ./redis-delkeys.sh [-h host] [-p port] [-n db] "pattern"
#
# Matches keys with the KEYS command matching pattern
#   and deletes them from the specified Redis DB.

set -e

HOST="localhost"
PORT="6379"
DB="0"
while getopts "h:p:n:" opt; do
    case $opt in
        h)  HOST=$OPTARG;;
        p)  PORT=$OPTARG;;
        n)  DB=$OPTARG;;
        \?) echo "invalid option: -$OPTARG" >&2; exit 1;;
    esac
done
shift $(( $OPTIND -1 ))
PATTERN="$@"
if [ -z "$PATTERN" ]; then
    echo "pattern required" >&2
    exit 2
fi

redis-cli -h $HOST -p $PORT -n $DB EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 "$PATTERN"