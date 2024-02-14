#date: 2024-02-14T16:55:13Z
#url: https://api.github.com/gists/cf8f3a1e57a754299298f236921ba3b0
#owner: https://api.github.com/users/manelatun

#!/bin/bash

# Add Docker's official GPG key.
sudo apt update
sudo apt install ca-certificates curl -y
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker's repository to Apt sources.
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update

# Install Docker's packages.
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# Grant the user permissions.
sudo groupadd docker
sudo usermod -aG docker $USER

echo 'Docker has been installed and your user has been granted permissions.'
