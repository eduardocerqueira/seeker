#date: 2025-05-23T16:57:25Z
#url: https://api.github.com/gists/4322c86dbcb0dace8393a56dd6624174
#owner: https://api.github.com/users/handeglc

#!/bin/bash

# Usage
# chmod +x migrate2github.sh
# ./migrate2github.sh -b https://bitbucket.org/your-user/your-repo.git -g https://github.com/your-user/your-repo.git

####################################

# Exit on any error
set -e

git config http.postBuffer 524288000

# Help message
usage() {
  echo "Usage: $0 -b <bitbucket-repo-url> -g <github-repo-url>"
  exit 1
}

# Parse flags
while getopts ":b:g:" opt; do
  case $opt in
    b) BITBUCKET_URL="$OPTARG"
    ;;
    g) GITHUB_URL="$OPTARG"
    ;;
    *) usage
    ;;
  esac
done

# Validate inputs
if [ -z "$BITBUCKET_URL" ] || [ -z "$GITHUB_URL" ]; then
  usage
fi

# Get repo name
REPO_NAME=$(basename -s .git "$BITBUCKET_URL")

# Clone as mirror
echo "Cloning Bitbucket repo as mirror..."
git clone --mirror "$BITBUCKET_URL"
cd "${REPO_NAME}.git"

# Add GitHub remote
echo "Adding GitHub remote..."
git remote add github "$GITHUB_URL"

# Push mirror to GitHub
echo "Pushing mirror to GitHub..."
git push --mirror github || echo "Mirror push failed, proceeding to check LFS..."

# Check if LFS files exist in mirror
echo "Checking for LFS files in Bitbucket clone..."
git lfs install

if git lfs ls-files | grep -q .; then
  echo "LFS files detected. Pushing to GitHub..."
  git lfs push --all github
else
  echo "No LFS files found."
fi

echo "Migration complete."
