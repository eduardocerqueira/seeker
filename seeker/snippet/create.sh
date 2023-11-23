#date: 2023-11-23T16:49:19Z
#url: https://api.github.com/gists/3c3eabddb392948fb0fefd027349440e
#owner: https://api.github.com/users/nsc-de

#!/bin/bash

# Git Repository creation script

# Settings

GITHUB_USERNAME={INSERT_YOUR_GITHUB_USERNAME_HERE}
PROVIDER_GIT=https://github.com/FOP-2324
REPO_PREFIX=TU-
REPO_DIR_PREFIX=TU-

# The repository name
REPO_NAME=$1
MY_REPO_NAME=$REPO_PREFIX$REPO_NAME
REPO_DIR=$REPO_DIR_PREFIX$REPO_NAME
MY_REPO=https://github.com/$GITHUB_USERNAME/$MY_REPO_NAME.git


# Create a new repository
if [ -z "$1" ]; then
    echo "Repository name is required"
    exit 1
fi

# Clone from github
if [ ! -d "$REPO_DIR" ]; then
    git clone $PROVIDER_GIT/$REPO_NAME.git $REPO_DIR
fi

cd $REPO_DIR

# Check if git repository exists (if not, create it) (using gh)
if ! gh repo view $MY_REPO_NAME > /dev/null 2>&1; then
    gh repo create $MY_REPO_NAME --confirm --private
    # Mirror the repository
    git push --mirror git@github.com:$GITHUB_USERNAME/$MY_REPO_NAME.git
fi

cd $REPO_DIR

# If "origin" remote exists and is not the same as the one we want, remove it
if git remote | grep -q origin && [ "$(git remote get-url origin)" != "$MY_REPO" ]; then
    git remote remove origin
fi

# Add "origin" remote
if ! git remote | grep -q origin; then
    git remote add origin $MY_REPO
fi

# Add "from" remote
if ! git remote | grep -q from; then
    git remote add from $PROVIDER_GIT/$REPO_NAME.git
fi

# If .github or workflows directory does not exist, create it
if [ ! -d ".github/workflows" ]; then
    mkdir -p .github/workflows
fi

# Copy workflow template
cp ../build-workflow-template.yml .github/workflows/build.yml

# Add and commit the workflow file if not present or has changes
if ! git ls-files --error-unmatch .github/workflows/build.yml > /dev/null 2>&1 || ! git diff --quiet .github/workflows/build.yml; then
    echo "Workflow file has changed or is not tracked. Adding and committing..."
    git add .github/workflows/build.yml
    git commit -m "Update build workflow"
    git push origin main
else
    echo "Workflow file has not changed"
fi