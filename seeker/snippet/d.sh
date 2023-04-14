#date: 2023-04-14T17:02:55Z
#url: https://api.github.com/gists/f095789e935ef6472e563a60974b648b
#owner: https://api.github.com/users/dafu

#!/bin/sh

# Docker installation Debia


apt-get remove docker docker-engine docker.io containerd runc

apt-get update

apt-get install \
    ca-certificates \
    curl \
    gnupg


install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update

apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin