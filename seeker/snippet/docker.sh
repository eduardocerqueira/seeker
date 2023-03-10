#date: 2023-03-10T16:53:19Z
#url: https://api.github.com/gists/b1abe6b45b617ae6d1a7334893ffc91a
#owner: https://api.github.com/users/retomi

#!/bin/bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=arm64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt -y install docker-ce
sudo apt -y install docker-compose
sudo usermod -aG docker www
