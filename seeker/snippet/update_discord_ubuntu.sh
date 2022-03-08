#date: 2022-03-08T17:04:25Z
#url: https://api.github.com/gists/9c9cb87e5b21396502cbf761a93583e9
#owner: https://api.github.com/users/ragazzid

#!/bin/bash
# Simple script to update discord :)

# Download
wget https://discord.com/api/download\?platform\=linux\&format\=deb -O /tmp/discord.deb

# Install
sudo dpkg -i /tmp/discord.dev

# Update 
sudo apt update

# Make sure all dependencies installed
sudo apt install -f -y

# Clean UP
rm -rf /tmp/discord.deb


# curl -L https://gist.githubusercontent.com/ragazzid/9c9cb87e5b21396502cbf761a93583e9/raw/bd6b412abb437288e7cab5fbaddc157639434924/update_discord_ubuntu.sh | bash
