#date: 2024-12-26T16:37:32Z
#url: https://api.github.com/gists/444a35de688d29c1a07a7f73a3c50b7e
#owner: https://api.github.com/users/natemollica-nm

#!/usr/bin/env zsh

# Function to print a colored message
info() {
  local color=$1
  local message=$2
  case $color in
    green) echo "\033[32m${message}\033[0m" ;;
    yellow) echo "\033[33m${message}\033[0m" ;;
    red) echo "\033[31m${message}\033[0m" ;;
    *) echo "$message" ;;
  esac
}

# Verify this is a Git repository
if [[ ! -d .git ]]; then
  info red "Error: This is not a Git repository."
  exit 1
fi

# Display current remote URLs
info yellow "Current remote configuration:"
git remote -v

# Prompt user for confirmation to proceed
printf '%s' "Do you want to update the remote URL for this repository? (y/n): "
read -r confirm </dev/tty
echo "" # Print a newline because -q does not automatically output one
if [[ "$confirm" != "y" ]]; then
  info red "Operation cancelled."
  exit 0
fi

# Ask for the new organization and repository name
read "org_name?Enter the new organization name: "
read "repo_name?Enter the new repository name: "

# Prompt for HTTPS or SSH
printf '%s' "Use HTTPS or SSH for the remote? (https/ssh): "
read -r remote_type </dev/tty
if [[ "$remote_type" == "https" ]]; then
  new_url="https://github.com/${org_name}/${repo_name}.git"
elif [[ "$remote_type" == "ssh" ]]; then
  new_url="git@github.com:${org_name}/${repo_name}.git"
else
  info red "Invalid choice. Please select 'https' or 'ssh'."
  exit 1
fi

# Update the origin remote
info yellow "Updating the 'origin' remote to: $new_url"
git remote set-url origin "$new_url"

# Verify the update
info yellow "Verifying the remote URL update..."
git remote -v

# Test the connection
info yellow "Testing connection to the new remote..."
if git fetch origin; then
  info green "Remote URL updated and verified successfully!"
else
  info red "Failed to connect to the new remote. Please check the URL and try again."
fi

# Optional: Ask if the user wants to rename the local repository folder
printf '%s' "Do you want to rename the local folder to match the new repository name? (y/n): "
read -r rename_confirm </dev/tty
echo "" # Print a newline because -q does not automatically output one
if [[ "$rename_confirm" == "y" ]]; then
  current_dir=$(basename "$PWD")
  cd ..
  mv "$current_dir" "$repo_name"
  cd "$repo_name"
  info green "Local folder renamed to $repo_name."
else
  info yellow "Local folder rename skipped."
fi

info green "Done!"
