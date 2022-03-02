#date: 2022-03-02T17:13:47Z
#url: https://api.github.com/gists/83229c0eaaa1d259a569cfb57ab75230
#owner: https://api.github.com/users/antl31

#!/bin/bash
sudo apt-get update


sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y



curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg


echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


sudo apt-get update


sudo apt-get install docker-ce docker-ce-cli containerd.io -y


sudo apt install screen -y
screen