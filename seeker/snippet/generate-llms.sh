#date: 2024-11-28T17:05:15Z
#url: https://api.github.com/gists/e1cf9bf4bdd8e16fb944beee7a6a746a
#owner: https://api.github.com/users/JayThibs

#!/bin/bash

# For usage in terminal:
# chmod +x generate-llms.sh
# sudo mv generate-llms.sh /usr/local/bin/generate-llms
# current directory: `generate-llms`
# Specified directory: `generate-llms [dir_name]`

# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [directory_path]"
    echo "If no directory is specified, uses current directory"
}

# Get directory path from argument or use current directory
DIR_PATH="${1:-.}"

# Validate directory exists
if [ ! -d "$DIR_PATH" ]; then
    echo "Error: Directory '$DIR_PATH' does not exist"
    print_usage
    exit 1
fi

# Get directory name for the title
DIR_NAME=$(basename "$(realpath "$DIR_PATH")")

# Create or overwrite llms.txt
{
    # Add title
    echo "# $DIR_NAME"
    echo

    # Add description from index.md if it exists
    if [ -f "$DIR_PATH/index.md" ]; then
        DESCRIPTION=$(awk '!/^#/ && NF {print; exit}' "$DIR_PATH/index.md")
    else
        DESCRIPTION="Documentation and resources for $DIR_NAME"
    fi
    echo "> $DESCRIPTION"
    echo

    # Add details about the repository
    echo "This documentation contains the following content:"
    echo

    # Find and include all markdown files
    find "$DIR_PATH" -type f \( -name "*.md" -o -name "*.markdown" \) ! -path "*/\.*" ! -name "llms.txt" | while read -r file; do
        echo "## File: ${file#"$DIR_PATH/"}"
        echo
        cat "$file"
        echo
        echo "---"
        echo
    done

} > "$DIR_PATH/llms.txt"

echo "Generated llms.txt in $DIR_PATH"