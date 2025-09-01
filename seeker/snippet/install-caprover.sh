#date: 2025-09-01T17:04:05Z
#url: https://api.github.com/gists/13fcc7abf719153298ce3f815f2214f7
#owner: https://api.github.com/users/austinsonger

#!/bin/bash

# Ubuntu 22.04
#   Please also allow `80, 443, 3000` ports in the VM network rules if apply

# run as sudo
if [ "$EUID" -ne 0 ]
  then echo "Please run as root or use sudo"
  exit
fi

# prepare installing docker
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# install docker engine
apt-get update
apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# update built-in firewall
# ufw allow 80,443,3000,996,7946,4789,2377/tcp
# ufw allow 7946,4789,2377/udp

# install CapRover
docker run -p 80:80 -p 443:443 -p 3000:3000 -v /var/run/docker.sock:/var/run/docker.sock -v /captain:/captain caprover/caprover

echo 'Please run "caprover serversetup" on your laptop to finish setup'
echo "IP: `hostname -I`"