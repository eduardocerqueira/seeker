#date: 2025-05-12T16:44:05Z
#url: https://api.github.com/gists/4b4086647b7989c90fabb9f0f290fd0e
#owner: https://api.github.com/users/matipojo

#!/bin/bash

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --source-a=*)
      REPO_A="${1#*=}"
      ;;
    --source-b=*)
      REPO_B="${1#*=}"
      ;;
    --target=*)
      TARGET_DIR="${1#*=}"
      ;;
    --folder-a=*)
      FOLDER_A="${1#*=}"
      ;;
    --folder-b=*)
      FOLDER_B="${1#*=}"
      ;;
    --branch-a=*)
      BRANCH_A="${1#*=}"
      ;;
    --branch-b=*)
      BRANCH_B="${1#*=}"
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# Set default branches if not specified
BRANCH_A=${BRANCH_A:-"main"}
BRANCH_B=${BRANCH_B:-"main"}

# Validate required parameters
if [ -z "$REPO_A" ] || [ -z "$REPO_B" ] || [ -z "$TARGET_DIR" ] || [ -z "$FOLDER_A" ] || [ -z "$FOLDER_B" ]; then
    echo "Usage: $0 --source-a=<repo-a-url> --source-b=<repo-b-url> --target=<target-dir> --folder-a=<folder-a> --folder-b=<folder-b> [--branch-a=<branch-a>] [--branch-b=<branch-b>]"
    echo "Example: $0 --source-a=https://github.com/org/repo-a.git --source-b=https://github.com/org/repo-b.git --target=./merged-repo --folder-a=project-a --folder-b=project-b --branch-a=main --branch-b=develop"
    exit 1
fi

# Create and enter target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

echo "Initializing new repository..."
git init

# Clone and prepare repo A
echo "Processing repository A ($REPO_A) branch $BRANCH_A..."
git remote add origin-a "$REPO_A"
git fetch origin-a
git checkout -b repo-a "origin-a/$BRANCH_A"

# Rewrite history with git filter-repo
echo "Rewriting history into subdirectory: $FOLDER_A/"
git filter-repo --to-subdirectory-filter "$FOLDER_A"

# Move files to ensure working tree consistency
mkdir -p "$FOLDER_A"
find . -maxdepth 1 ! -name "$FOLDER_A" ! -name "$FOLDER_B" ! -name ".git" ! -name "." -exec git mv {} "$FOLDER_A/" \; 2>/dev/null || true
git add "$FOLDER_A"
git commit -m "Restructure repo A into $FOLDER_A/"

# Clone and prepare repo B
echo "Processing repository B ($REPO_B) branch $BRANCH_B..."
git remote add origin-b "$REPO_B"
git fetch origin-b
git checkout -b repo-b "origin-b/$BRANCH_B"

# Rewrite history with git filter-repo
echo "Rewriting history into subdirectory: $FOLDER_B/"
git filter-repo --to-subdirectory-filter "$FOLDER_B"

# Move files to ensure working tree consistency
mkdir -p "$FOLDER_B"
find . -maxdepth 1 ! -name "$FOLDER_A" ! -name "$FOLDER_B" ! -name ".git" ! -name "." -exec git mv {} "$FOLDER_B/" \; 2>/dev/null || true
git add "$FOLDER_B"
git commit -m "Restructure repo B into $FOLDER_B/"

# Create main branch and merge both repositories
echo "Merging repositories..."
git checkout -b main repo-a
git merge repo-b --allow-unrelated-histories -m "Merge repository B into A, preserving history"

# Clean up
git branch -D repo-a repo-b
git remote remove origin-a
git remote remove origin-b

echo "✅ Repositories merged successfully into $TARGET_DIR"
echo "📁 Repo A in folder: $FOLDER_A (from branch $BRANCH_A)"
echo "📁 Repo B in folder: $FOLDER_B (from branch $BRANCH_B)"
echo "🔍 Full history has been preserved"