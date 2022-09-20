#date: 2022-09-20T17:11:14Z
#url: https://api.github.com/gists/c116baaa4cabcf4b00286739c3e297b0
#owner: https://api.github.com/users/Curicows

#!/bin/sh

# Shell script to add docker-ce to Deepin Linux repositories

# Remove old docker
sudo apt-get remove -y docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg2 software-properties-common
    
# Add GPG and fingerprint
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

# Add debian repository in deepin sources list
printf 'deb [arch=amd64] https://download.docker.com/linux/debian buster stable\n' \
| sudo tee /etc/apt/sources.list.d/docker-ce.list

# Update packages and install
sudo apt-get update -y
sudo apt-get install -y docker-ce

# Add current user to the docker users group (OPTIONAL)
sudo usermod -aG docker $(whoami)