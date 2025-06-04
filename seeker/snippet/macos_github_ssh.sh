#date: 2025-06-04T17:11:41Z
#url: https://api.github.com/gists/3567eb9dbefadb041413109dfa86a416
#owner: https://api.github.com/users/arec1b0

#!/bin/bash

# --- Script to Create SSH Key and Configure for GitHub on macOS ---

echo "GitHub SSH Key Setup for macOS"
echo "---------------------------------"

# 1. Ask for GitHub email
read -p "Enter your GitHub email address: " GITHUB_EMAIL

if [ -z "$GITHUB_EMAIL" ]; then
  echo "Email address cannot be empty. Exiting."
  exit 1
fi

KEY_FILENAME="id_ed25519_github_$(date +%Y%m%d%H%M%S)"
SSH_DIR="$HOME/.ssh"
PRIVATE_KEY_PATH="$SSH_DIR/$KEY_FILENAME"
PUBLIC_KEY_PATH="$SSH_DIR/$KEY_FILENAME.pub"
CONFIG_FILE="$SSH_DIR/config"

echo ""
echo "This script will generate a new Ed25519 SSH key."
echo "You will be prompted to enter a passphrase for the key."
echo "It is STRONGLY recommended to use a passphrase."
echo ""

# 2. Generate a new Ed25519 SSH key
echo "Generating SSH key..."
# The -f flag specifies the filename, -N "" would set an empty passphrase (not recommended)
ssh-keygen -t ed25519 -C "$GITHUB_EMAIL" -f "$PRIVATE_KEY_PATH"

if [ $? -ne 0 ]; then
  echo "SSH key generation failed. Exiting."
  exit 1
fi

echo "SSH key generated at $PRIVATE_KEY_PATH and $PUBLIC_KEY_PATH"
echo ""

# 3. Ensure ssh-agent is running
echo "Ensuring ssh-agent is running..."
eval "$(ssh-agent -s)"
echo ""

# 4. Create/Update ~/.ssh/config file
echo "Configuring $CONFIG_FILE..."
mkdir -p "$SSH_DIR" # Ensure .ssh directory exists
touch "$CONFIG_FILE" # Ensure config file exists
chmod 600 "$CONFIG_FILE" # Set correct permissions

# Check if an entry for github.com with this key already exists
if ! grep -q "Host github.com" "$CONFIG_FILE" || ! grep -q "IdentityFile $PRIVATE_KEY_PATH" "$CONFIG_FILE"; then
  # Add new configuration, or update if Host github.com exists but for a different key
  # This basic version just appends; more sophisticated merging could be done
  {
    echo "" # Add a newline for separation
    echo "Host github.com"
    echo "  HostName github.com"
    echo "  User git"
    echo "  AddKeysToAgent yes"
    echo "  UseKeychain yes" # macOS specific, stores passphrase in Keychain
    echo "  IdentityFile $PRIVATE_KEY_PATH"
    echo "  IdentitiesOnly yes" # Prevents SSH from trying default key names first
  } >> "$CONFIG_FILE"
  echo "GitHub configuration added to $CONFIG_FILE for key $KEY_FILENAME"
else
  echo "GitHub configuration for this key already seems to exist or Host github.com points to a different key in $CONFIG_FILE. Please check manually."
fi
echo ""

# 5. Add SSH private key to ssh-agent and macOS Keychain
echo "Adding SSH private key to ssh-agent and Keychain..."
echo "You might be prompted for the passphrase you just created, and then for your macOS login password to store it in the Keychain."
ssh-add --apple-use-keychain "$PRIVATE_KEY_PATH" # For macOS. On Linux, it's just `ssh-add <path>`

if [ $? -ne 0 ]; then
  echo "Failed to add SSH key to agent. You may need to add it manually:"
  echo "  ssh-add --apple-use-keychain $PRIVATE_KEY_PATH"
  echo "If it asks for a passphrase, enter the one you set for the key."
fi
echo ""

# 6. Copy the public key to clipboard
echo "Copying public key ($PUBLIC_KEY_PATH) to clipboard..."
pbcopy < "$PUBLIC_KEY_PATH"
echo "Your new public SSH key has been copied to the clipboard!"
echo ""

# 7. Instructions for GitHub
echo "---------------------------------"
echo "NEXT STEPS: Add the SSH Key to GitHub"
echo "---------------------------------"
echo "1. Go to GitHub.com and log in."
echo "2. Click on your profile picture in the top-right corner, then click 'Settings'."
echo "3. In the user settings sidebar, click 'SSH and GPG keys'."
echo "4. Click 'New SSH key' or 'Add SSH key'."
echo "5. In the 'Title' field, add a descriptive label (e.g., \"MacBook Air $(date +%Y-%m-%d)\" or \"Personal Laptop\")."
echo "6. Paste your SSH public key (which is now on your clipboard) into the 'Key' field."
echo "7. Click 'Add SSH key'."
echo "You may be asked to confirm your GitHub password."
echo ""

# 8. Test the SSH connection
echo "---------------------------------"
echo "After adding the key to GitHub, you can test your connection:"
echo "Run this command in your terminal:"
echo "  ssh -T git@github.com"
echo ""
echo "You should see a message like: 'Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.'"
echo "If it's the first time connecting, you might see a host authenticity warning, type 'yes'."
echo "---------------------------------"
echo "Script finished."