#date: 2025-07-17T17:03:18Z
#url: https://api.github.com/gists/868417a621bc7d189ad7f92877a12327
#owner: https://api.github.com/users/panchishin

#!/usr/bin/env bash
# Usage: some_command | mytail.sh [N]
LINES=${1:-10} # Default to 10, or use user-specified number

# Array to hold the last N lines
declare -a buffer

while IFS= read -r line; do
    buffer+=("$line")
    # Keep buffer at $LINES
    if [ ${#buffer[@]} -gt "$LINES" ]; then
        buffer=("${buffer[@]:1}")
    fi

    # Move cursor up if necessary
    [ "$printed" = true ] && printf '\e[%sA' "$LINES"

    # Print stored lines, pad with blanks if not enough
    pad=$((LINES - ${#buffer[@]}))
    for ((i=0;i<pad;i++)); do
        printf "\e[2K\n"
    done
    for x in "${buffer[@]}"; do
        printf "\e[2K"
        printf "%s\n" "$x"
    done

    printed=true
done

# At script end, cursor will stay at the last printed line.
