#date: 2023-02-17T16:55:08Z
#url: https://api.github.com/gists/af1472c602736e83a23186c3949ba4f0
#owner: https://api.github.com/users/butlerbt

#!/bin/bash

LOG_FILE=/tmp/memory_usage.log
touch $LOG_FILE

while true; do
    # Print the date and time to the log file
    echo "=== $(date +'%Y-%m-%d %H:%M:%S') ===" >> $LOG_FILE

    # Sort the array by memory usage and print the top 10 results to the log file
    echo "$(ps ax -m -o %mem,command | grep -ve '- -' | sort -rn | head -n 10)" >> $LOG_FILE

    # Sleep for 5 minutes before running again
    sleep 300
done