#date: 2025-07-28T16:57:38Z
#url: https://api.github.com/gists/70711501810dc4c39ccb6d723ef3f85e
#owner: https://api.github.com/users/MykalMachon

#!/usr/bin/env sh

# Function to display usage
usage() {
    echo "Usage: $0 <size>"
    echo "  size: "**********"
    echo "Example: $0 32"
    exit 1
}

# Check if argument is provided
if [ $# -gt 1 ]; then
    echo "Error: Exactly one argument required"
    usage
elif [ $# -eq 0 ]; then
	# set default size of 24
	set -- "24"
fi

# Validate that the argument is a number (POSIX-compliant)
if [ $# -eq 1 ]; then
    case "$1" in
        ''|*[!0-9]*) 
            echo "Error: '$1' is not a valid number"
            usage
            ;;
    esac
fi

# Store the size
SIZE="$1"

# Validate reasonable bounds (optional - adjust as needed)
if [ "$SIZE" -lt 1 ] || [ "$SIZE" -gt 1000 ]; then
    echo "Error: Size must be between 1 and 1000"
    exit 1
fi

# Fetch the secret
echo "Fetching secret of size $SIZE..."
if command -v curl >/dev/null 2>&1; then
    SECRET=$(curl -s "https: "**********"
elif command -v wget >/dev/null 2>&1; then
    SECRET=$(wget -qO- "https: "**********"
else
    echo "Error: Neither curl nor wget is available"
    exit 1
fi

# Check if secret was fetched successfully
if [ -z "$SECRET" ]; then
    echo "Error: "**********"
    exit 1
fi

# Display the secret
echo "$SECRET"

# Attempt to copy to clipboard
copy_to_clipboard() {
    if command -v pbcopy >/dev/null 2>&1; then
        echo "$SECRET" | pbcopy
        echo "Secret copied to clipboard (macOS)"
        return 0
    elif command -v xclip >/dev/null 2>&1; then
        echo "$SECRET" | xclip -selection clipboard
        echo "Secret copied to clipboard (Linux - xclip)"
        return 0
    elif command -v xsel >/dev/null 2>&1; then
        echo "$SECRET" | xsel --clipboard --input
        echo "Secret copied to clipboard (Linux - xsel)"
        return 0
    elif command -v wl-copy >/dev/null 2>&1; then
        echo "$SECRET" | wl-copy
        echo "Secret copied to clipboard (Wayland)"
        return 0
    elif command -v clip.exe >/dev/null 2>&1; then
        echo "$SECRET" | clip.exe
        echo "Secret copied to clipboard (WSL)"
        return 0
    else
        echo "Note: "**********"
        return 1
    fi
}

copy_to_clipboard        echo "Note: No clipboard utility found. Secret not copied to clipboard."
        return 1
    fi
}

copy_to_clipboard