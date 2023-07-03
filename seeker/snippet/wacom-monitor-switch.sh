#date: 2023-07-03T17:02:02Z
#url: https://api.github.com/gists/617bc10138b76c376c3828dc43315fe4
#owner: https://api.github.com/users/Lucs1590

#!/bin/bash

# set -euo pipefail

# Check if monitor number was provided as an argument
if [[ $# -eq 0 ]]; then
    echo "Please provide the monitor number as an argument."
    echo "Usage: $0 <monitor_number>"
    exit 1
fi

# Extract the monitor number from the argument
MON_NO=$1

# Get the connection of the monitor with number $MON_NO
MON_CON=$(xrandr --listactivemonitors | grep "$MON_NO:" | awk '{print $NF}')

# Check if monitor ID could be found
if [[ -z $MON_CON ]]; then
    # No monitor with the given number found
    echo "No monitor with number $MON_NO was found!"
    MON_COUNT=$(xrandr --listactivemonitors | grep "Monitors:" | awk '{print $2}')
    echo "(A total of $MON_COUNT monitors were found.)"
    exit 1
fi

# Set the correct output monitor for each Wacom device
xsetwacom list devices |
    while IFS= read -r line; do
        # Get the ID of the Wacom device in the line
        # e.g. " Wacom One by Wacom S Pen stylus         id: 27  type: STYLUS"
        WACOM_ID=$(echo "$line" | awk -F '[: ]' '{print $4}')
        # Set the monitor as output
        xsetwacom --set "$WACOM_ID" MapToOutput "$MON_CON"
    done

echo "Changed the monitor for all Wacom devices."
exit 0
