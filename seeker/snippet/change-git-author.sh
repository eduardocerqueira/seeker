#date: 2025-04-14T17:02:23Z
#url: https://api.github.com/gists/be08ec7820543e0696f0ff1129b2c1b9
#owner: https://api.github.com/users/angelmmg90

#!/bin/bash

# Script to change the author of all commits in a Git repository

# Colors for messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}This script will change the author on ALL commits in this repository.${NC}"
echo -e "${RED}WARNING! This operation rewrites Git history and changes commit hashes.${NC}"
echo -e "${RED}If you have already shared this repository with others, this could cause problems.${NC}"
echo ""

# Request current author information
read -p "Enter the current author name (press Enter to use 'angel.macdonald'): " OLD_NAME
OLD_NAME=${OLD_NAME:-"angel.macdonald"}

read -p "Enter the current author email (press Enter to use 'angel.macdonald@sngular.com'): " OLD_EMAIL
OLD_EMAIL=${OLD_EMAIL:-"angel.macdonald@sngular.com"}

# Request new author information
read -p "Enter the new author name: " NEW_NAME
while [ -z "$NEW_NAME" ]; do
    echo -e "${RED}Error: Author name cannot be empty.${NC}"
    read -p "Enter the new author name: " NEW_NAME
done

read -p "Enter the new author email: " NEW_EMAIL
while [ -z "$NEW_EMAIL" ]; do
    echo -e "${RED}Error: Author email cannot be empty.${NC}"
    read -p "Enter the new author email: " NEW_EMAIL
done

# Final confirmation
echo ""
echo -e "${YELLOW}You are about to change:${NC}"
echo -e "Current author: ${OLD_NAME} <${OLD_EMAIL}>"
echo -e "New author: ${NEW_NAME} <${NEW_EMAIL}>"
echo ""
read -p "Are you sure you want to continue? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

# Create a filter to rewrite history
git filter-branch --env-filter "
if [ \"\$GIT_COMMITTER_NAME\" = \"$OLD_NAME\" ] || [ \"\$GIT_COMMITTER_EMAIL\" = \"$OLD_EMAIL\" ]; then
    export GIT_COMMITTER_NAME=\"$NEW_NAME\"
    export GIT_COMMITTER_EMAIL=\"$NEW_EMAIL\"
fi
if [ \"\$GIT_AUTHOR_NAME\" = \"$OLD_NAME\" ] || [ \"\$GIT_AUTHOR_EMAIL\" = \"$OLD_EMAIL\" ]; then
    export GIT_AUTHOR_NAME=\"$NEW_NAME\"
    export GIT_AUTHOR_EMAIL=\"$NEW_EMAIL\"
fi
" --tag-name-filter cat -- --branches --tags

# Result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Author change completed successfully!${NC}"
    echo -e "${YELLOW}Important notes:${NC}"
    echo "1. Commit hashes have changed."
    echo "2. If this repository was already shared/published, you'll need to 'git push --force'."
    echo "3. People who are collaborating will need to clone the repository again or handle this situation."
    echo ""
    echo -e "${YELLOW}To confirm changes to a remote repository:${NC}"
    echo "git push --force --all"
    echo "git push --force --tags"
else
    echo -e "${RED}An error occurred while changing the author.${NC}"
    echo "Check for error messages above and fix them."
fi