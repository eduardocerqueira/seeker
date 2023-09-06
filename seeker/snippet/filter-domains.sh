#date: 2023-09-06T16:58:00Z
#url: https://api.github.com/gists/b2b9f9d16146b6d5e3ab6b3553b8febf
#owner: https://api.github.com/users/vlazic

#!/bin/bash

DOMAINS_FILE="domains.txt"
OUTPUT_FILE="domains_with_header.txt"
PROGRESS_FILE="progress.txt"

# Function to print the current status
print_status() {
    echo "Total domains: $total_lines"
    echo "Testing domain ($current_line/$total_lines): $domain"
}

# Trap function to save progress on interrupt
handle_interrupt() {
    echo $current_line >"$PROGRESS_FILE"
    echo "Saved progress at line $current_line. Rerun script to continue."
    exit 1
}

trap 'handle_interrupt' INT

# If there's an argument provided, start from that line. Otherwise, start from the beginning or from saved progress.
if [[ $1 ]]; then
    start_line=$1
elif [[ -f $PROGRESS_FILE ]]; then
    start_line=$(cat "$PROGRESS_FILE")
else
    start_line=1
fi

total_lines=$(wc -l <"$DOMAINS_FILE")
current_line=$start_line

# Loop through domains starting from the specified or saved line
sed -n "${start_line},$ p" "$DOMAINS_FILE" | while read -r domain; do
    print_status
    response=$(curl --max-time 10 -Is "https://$domain" | grep 'X-Booking-Engine')
    if [[ ! -z $response ]]; then
        echo $domain >>"$OUTPUT_FILE"
    fi
    ((current_line++))
done

# Clean up progress file if script completes successfully
rm -f "$PROGRESS_FILE"
