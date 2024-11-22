#date: 2024-11-22T17:12:25Z
#url: https://api.github.com/gists/fd5ee4a1aba87382946adb1a8fb1a818
#owner: https://api.github.com/users/devnoot

#!/bin/env bash

# Create the .ssh directory if it does not exist.
if [ ! -d ~/.ssh ]; then
  mkdir -p ~/.ssh
fi

# Create the .ssh config file if it   does not exist.
if [ ! -f ~/.ssh/config ]; then
  touch ~/.ssh/config
fi

# Set the correct permissions for ssh directory and keys
chmod 700 ~/.ssh
find ~/.ssh -type f -name "*.pub" -exec chmod 644 {} \;
find ~/.ssh \( -type f \( -name "*.pem" -o ! -name "*.*" \) \) -exec chmod 600 {} \;
chmod 600 ~/.ssh/config

# Confirmation message
echo "SSH directory and file permissions have been updated."
