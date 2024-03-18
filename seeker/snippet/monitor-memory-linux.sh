#date: 2024-03-18T17:04:16Z
#url: https://api.github.com/gists/8993168ceb2ad6340161348e8f082faa
#owner: https://api.github.com/users/santhoshsram

#!/bin/bash

# Initialize variables with default values
pid=""
interval=5
max_rss=0

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pid) pid="$2"; shift ;;
        --interval) interval="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required arguments are provided
if [ -z "$pid" ]; then
    echo "Usage: $0 --pid <pid> [--interval <seconds>]"
    exit 1
fi

# Infinite loop
while true; do
    # Get VmRSS value from /proc/<pid>/status
    rss=$(awk '/VmRSS/ {print $2}' "/proc/$pid/status")

    # Update max_rss if needed
    if (( rss > max_rss )); then
        max_rss=$rss
    fi

    # Convert to appropriate units
    if (( max_rss < 1024 )); then
        units="KB"
        value="$max_rss"
    elif (( max_rss < 1048576 )); then
        units="MB"
        value=$(bc <<< "scale=2; $max_rss / 1024")
    else
        units="GB"
        value=$(bc <<< "scale=2; $max_rss / 1048576")
    fi

    # Print max_rss with units
    printf "\rMax Memory: %.2f %s" "$value" "$units"

    # Sleep for the specified interval
    sleep "$interval"
done

