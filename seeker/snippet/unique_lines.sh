#date: 2025-07-25T17:14:34Z
#url: https://api.github.com/gists/2c913f0f2c72c66b6b24b7fb402adee3
#owner: https://api.github.com/users/mfortuno-keap

#!/bin/bash

# Check for two arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <file1> <file2>"
    exit 1
fi

# Verify files exist
if [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "Error: Both files must exist"
    exit 1
fi

# Output lines unique to file1
comm -23 <(sort "$1") <(sort "$2")