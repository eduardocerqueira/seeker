#date: 2025-02-17T16:41:32Z
#url: https://api.github.com/gists/a1c353aa053ae3b61a541f38e229b05e
#owner: https://api.github.com/users/bkataru

#!/bin/bash

# Initialize the context string
CONTEXT=""

# Find all files, excluding specified directories and self/output
while IFS= read -r -d '' file; do
    # Get file metadata
    REL_PATH="${file#./}"
    
    # Append file metadata and content to context
    CONTEXT+="==== FILE: $REL_PATH ===="$'\n'
    CONTEXT+="$(cat "$file")"$'\n\n'
    
done < <(find . -type d \( -path "./config" -o -path "./data" -o -path "./.git" -o -path "./RTSPtoHLS" \) -prune -o -type f \( -not -name "generate_context.sh" -a -not -name "context.txt" \) -print0)

# Output the compiled context
echo "$CONTEXT"
