#date: 2025-03-05T17:03:38Z
#url: https://api.github.com/gists/5fc9d8621bb3d6a346ccc1d139701f37
#owner: https://api.github.com/users/jerodg

#!/bin/bash
# filepath: github_autopush.sh

###############################################################################
# GitHub Auto-Push Script
# Author: JerodG https://github.com/jerodg
#
# This script automates the process of pushing local git repositories to GitHub.
# It scans the current directory for git repositories, creates private GitHub
# repositories for each one, and pushes all branches and tags.
#
# Dependencies:
#   - GitHub CLI (gh) must be installed (https://cli.github.com/)
#   - jq must be installed for JSON parsing (https://jqlang.org/)
#   - User must have permissions to create repositories
#
# Security Notice:
#   The script should use environment variables or a secure credential store
#   rather than hardcoding tokens directly in the script.
###############################################################################

# SECURITY ISSUE: "**********"
# Use environment variables instead, e.g., GH_TOKEN= "**********"
GH_TOKEN= "**********"

# Verifies the GitHub CLI is available as it's required for all operations
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI not found. Please install it from https://cli.github.com/"
    exit 1
fi

# Authenticates with GitHub using the provided token to avoid interactive prompts
# SSH protocol is selected for git operations for better security
gh auth login --with-token $GH_TOKEN --git-protocol ssh

# Verifies authentication was successful before proceeding
if ! gh auth status &> /dev/null; then
    echo "Error: You are not authenticated with GitHub CLI. Please run 'gh auth login' first."
    exit 1
fi

###############################################################################
# Creates a GitHub repository and pushes local content to it
#
# Handles creation of GitHub repository, remote configuration, and pushing
# all branches and tags. Preserves existing remotes when present.
#
# Arguments:
#   $1 - Full path to the local git repository
#   $2 - Repository name to be created on GitHub
#
# Returns:
#   0 if successful, 1 if any step fails
#
# Side effects:
#   - Creates a new GitHub repository
#   - Adds or modifies git remotes in the local repository
#   - Pushes content to GitHub
###############################################################################
create_and_push() {
    local repo_path="$1"
    local repo_name="$2"
    
    echo "-----------------------------------------------------"
    echo "Processing: $repo_name ($repo_path)"
    
    # Changes to repository directory to perform git operations
    cd "$repo_path" || return
    
    # Prevents accidental duplication by checking if the repository is already
    # connected to GitHub and prompting for confirmation
    if git remote -v | grep -q "github.com"; then
        echo "This repository appears to be already connected to GitHub."
        read -p "Do you want to create a new repository anyway? (y/n): " proceed
        if [[ "$proceed" != [Yy]* ]]; then
            echo "Skipping this repository."
            cd - > /dev/null
            return 0
        fi
    fi
    
    # Creates a private repository on GitHub with the same name as the local directory
    echo "Creating private GitHub repository: $repo_name"
    if ! gh repo create "$repo_name" --private --confirm; then
        echo "Error: Failed to create GitHub repository for $repo_name."
        cd - > /dev/null
        return 1
    fi
    
    # Retrieves the authenticated user's GitHub username for remote URL construction
    username=$(gh api user | jq -r .login)
    if [[ -z "$username" || "$username" == "null" ]]; then
        echo "Error: Failed to retrieve GitHub username."
        cd - > /dev/null
        return 1
    fi
    
    # Preserves existing remotes by using a different remote name if origin exists
    if git remote get-url origin &> /dev/null; then
        echo "Remote 'origin' already exists. Adding 'github' as a new remote."
        git remote add github "https://github.com/$username/$repo_name.git"
        remote_name="github"
    else
        echo "Adding GitHub as the 'origin' remote."
        git remote add origin "https://github.com/$username/$repo_name.git"
        remote_name="origin"
    fi
    
    # Pushes all branches to ensure complete repository migration
    echo "Pushing all branches to GitHub..."
    if ! git push -u "$remote_name" --all; then
        echo "Error: Failed to push branches to GitHub."
        cd - > /dev/null
        return 1
    fi
    
    # Ensures tags are also pushed as they're not included in the --all flag
    echo "Pushing all tags to GitHub..."
    git push "$remote_name" --tags
    
    echo "Success: Repository $repo_name successfully pushed to GitHub."
    cd - > /dev/null
}

echo "Finding git repositories in the current directory..."

# Stores paths to discovered git repositories
declare -a repo_paths

# Identifies repositories by locating .git directories and extracting their parent paths
while IFS= read -r gitdir; do
    repo_path=$(dirname "$gitdir")
    # Converts to absolute path to avoid relative path issues during directory changes
    cd "$repo_path" || continue
    abs_path=$(pwd)
    cd - > /dev/null
    repo_paths+=("$abs_path")
done < <(find . -type d -name ".git")

# Filters out nested repositories to prevent duplicate processing and potential conflicts
# This is important for monorepos or projects that contain other git repositories
declare -a filtered_repos
for repo in "${repo_paths[@]}"; do
    is_nested=false
    for other_repo in "${repo_paths[@]}"; do
        if [[ "$repo" != "$other_repo" && "$repo" == "$other_repo"/* ]]; then
            is_nested=true
            echo "Skipping nested repository: $repo (inside $other_repo)"
            break
        fi
    done
    
    if [[ "$is_nested" == false ]]; then
        filtered_repos+=("$repo")
    fi
done

repo_count=${#filtered_repos[@]}

# Early termination if no repositories are found
if [ "$repo_count" -eq 0 ]; then
    echo "No git repositories found in the current directory."
    exit 0
fi

# Requests user confirmation before proceeding with repository creation
# to prevent unintentional creation of multiple repositories
echo "Found $repo_count git repositories."
read -p "Do you want to create private GitHub repositories for all of them? (y/n): " confirm
if [[ "$confirm" != [Yy]* ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Processes each repository individually to isolate failures
for repo_path in "${filtered_repos[@]}"; do
    repo_name=$(basename "$repo_path")
    create_and_push "$repo_path" "$repo_name"
done

echo "-----------------------------------------------------"
echo "All repositories processed."
positories processed."
