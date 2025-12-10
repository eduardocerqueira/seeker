#date: 2025-12-10T16:58:41Z
#url: https://api.github.com/gists/4b36ff5fead2407f8bd10842615e4c7b
#owner: https://api.github.com/users/schicks

#!/bin/bash

# Find all AGENTS.md files starting from current directory and create CLAUDE.md symlinks
find . -type f -name "AGENTS.md" | while read -r agents_file; do
    # Get the directory containing the AGENTS.md file
    dir=$(dirname "$agents_file")

    # Create symlink path
    claude_link="$dir/CLAUDE.md"

    # Check if CLAUDE.md already exists
    if [ -e "$claude_link" ]; then
        echo "Skipping: $claude_link already exists"
    else
        # Create relative symlink (just the filename since they're in same dir)
        ln -s "AGENTS.md" "$claude_link"
        echo "Created symlink: $claude_link -> AGENTS.md"
    fi
done

echo "Done creating CLAUDE.md symlinks"