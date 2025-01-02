#date: 2025-01-02T17:12:32Z
#url: https://api.github.com/gists/2818c32c4ae2c4754a9428d5a436e41c
#owner: https://api.github.com/users/viveknair

#!/bin/bash

# Script to analyze git LOC changes
# Usage: ./git-loc-changes.sh [options]
# Options:
#   -s, --since DATE    Show changes since DATE (e.g., "1 week ago", "2023-01-01")
#   -a, --author NAME   Filter by author
#   -h, --help         Show this help message

print_help() {
    grep "^#" "$0" | cut -c 3-
    exit 0
}

# Default values
SINCE=""
AUTHOR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--since)
            SINCE="$2"
            shift 2
            ;;
        -a|--author)
            AUTHOR="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

# Construct git log command based on parameters
GIT_CMD="git log --numstat"
if [ ! -z "$SINCE" ]; then
    GIT_CMD="$GIT_CMD --since=\"$SINCE\""
fi
if [ ! -z "$AUTHOR" ]; then
    GIT_CMD="$GIT_CMD --author=\"$AUTHOR\""
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not a git repository"
    exit 1
fi

echo "Computing LOC changes..."
echo "------------------------"

# Calculate total changes
eval "$GIT_CMD" | awk '
/^[0-9]/ {
    additions += $1
    deletions += $2
}
END {
    print "Total lines added:   " additions
    print "Total lines deleted: " deletions
    print "Net change:         " additions - deletions
}'

# Show changes by file extension
echo -e "\nChanges by file extension:"
echo "------------------------"
eval "$GIT_CMD" | awk '
/^[0-9]/ {
    split($3, parts, ".")
    if (length(parts) > 1) {
        ext = parts[length(parts)]
        adds[ext] += $1
        dels[ext] += $2
    }
}
END {
    for (ext in adds) {
        printf "%-10s +%-8d -%-8d (net: %d)\n", 
               ext, 
               adds[ext], 
               dels[ext], 
               adds[ext] - dels[ext]
    }
}' | sort -k2nr 