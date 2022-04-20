#date: 2022-04-20T17:01:03Z
#url: https://api.github.com/gists/8a103851bcf4a3e3fc88e7e67ed5c153
#owner: https://api.github.com/users/yeeyangtee

#!/bin/bash
sudo apt update && sudo apt upgrade
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo docker run hello-world
# Linux post-install
sudo groupadd docker
sudo usermod -aG docker $USER
sudo systemctl enable docker