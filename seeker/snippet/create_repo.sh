#date: 2025-02-05T16:50:20Z
#url: https://api.github.com/gists/e28b57e34dace656967c9888d90441a9
#owner: https://api.github.com/users/AlsGomes

#!/bin/bash

# Check if correct number of arguments is provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <directory> <username> <user-email> <commit-message> <github-remote-url>"
    exit 1
fi

# Check if GitHub token is set
if [ -z "$GITHUB_ACCESS_TOKEN" ]; then
    echo "Error: "**********"
    echo  "**********"Please set your GitHub personal access token first: "**********"
    echo "  export GITHUB_ACCESS_TOKEN= "**********"
    echo "You can generate this token at: "**********"://github.com/settings/tokens"
    exit 1
fi

# Assign arguments to variables
DIRECTORY=$1
USERNAME=$2
USEREMAIL=$3
COMMITMESSAGE=$4
REMOTEURL=$5

# Check if directory exists, create if not
mkdir -p "$DIRECTORY"

# Move to the directory
cd "$DIRECTORY" || exit

# Initialize git repository
git init

# Add all files
git add .

# Configure user details
git config user.name "$USERNAME"
git config user.email "$USEREMAIL"

# Commit changes
git commit -m "$COMMITMESSAGE"

# Rename default branch to main
git branch -M main

# Add remote origin with embedded token
TOKENIZED_REMOTE=$(echo "$REMOTEURL" | sed "s|https: "**********"://${USERNAME}:${GITHUB_ACCESS_TOKEN}@|")
git remote add origin "$TOKENIZED_REMOTE"

# Push to remote repository
git push -u origin main

echo "Repository created successfully in $DIRECTORY"ly in $DIRECTORY"