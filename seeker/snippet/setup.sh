#date: 2023-06-21T17:02:55Z
#url: https://api.github.com/gists/573e92fdcd47da930110c28ce4277cb4
#owner: https://api.github.com/users/pschur

#!/bin/bash

echo "Update Package List";
sudo apt update

echo "Upgrading system";
sudo apt upgrade -y

echo "Installing curl";
sudo apt install curl -y

echo "Installing Docker";
curl -fsSL https://get.docker.com | bash > ./docker.lock

echo "Installing Docker Compose"
sudo apt install docker-compose -y 

cat ./docker.lock
rm ./docker.lock

echo "Docker is installed";