#date: 2022-03-10T17:08:45Z
#url: https://api.github.com/gists/204d3e6ae6ea4da8d5a3a2c5f2bd8180
#owner: https://api.github.com/users/trung

#/bin/bash

sudo apt update
sudo apt -y upgrade
sudo apt -y install rsyslog openjdk-8-jre-headless rng-tools

sudo systemctl restart rng-tools
echo 'deb https://www.ui.com/downloads/unifi/debian stable ubiquiti' | sudo tee /etc/apt/sources.list.d/100-ubnt-unifi.list
sudo wget -O /etc/apt/trusted.gpg.d/unifi-repo.gpg https://dl.ui.com/unifi/unifi-repo.gpg
sudo apt update

sudo apt -y install unifi