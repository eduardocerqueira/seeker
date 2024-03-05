#date: 2024-03-05T17:07:33Z
#url: https://api.github.com/gists/e69deccd8f3edf22f528e636ff457f18
#owner: https://api.github.com/users/jcanofuentes

#!/bin/bash

# Install Git
sudo apt update && sudo apt install -y git

# Verify version
git --version

# Ask username and email
read -p "Enter your name for Git: " git_username
read -p "Enter your email address for Git: " git_email

# Configure Git with the name and email address provided
git config --global user.name "$git_username"
git config --global user.email "$git_email"

# Configure default brach
git config --global init.defaultBranch main

# Show current Git configuration
cat ~/.gitconfig

# Generate a key and assign it to the email
ssh-keygen -t rsa -b 4096 -C "$git_email" -N "" -f ~/.ssh/id_rsa

# In case you want to open the public key with VisualCode
#code ~/.ssh/id_rsa.pub

# Copy the content to the portapapeles
cat ~/.ssh/id_rsa.pub | clip.exe

echo "Git and SSH configuration completed."

