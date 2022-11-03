#date: 2022-11-03T17:06:29Z
#url: https://api.github.com/gists/95dae74dd8943596d1c8f8e873bc18c3
#owner: https://api.github.com/users/james-see

#!/bin/bash
# vars
VPNIP=127.0.0.1 # change this
# base
sudo ufw default deny outgoing
sudo ufw default deny incoming
sudo ufw allow ssh
sudo ufw reload
# github
sudo ufw allow to 185.199.108.0/22
sudo ufw allow to 140.82.112.0/20
sudo ufw allow to 143.55.64.0/20
sudo ufw allow to 2a0a:a440::/29
sudo ufw allow to 2606:50c0::/32
sudo ufw allow to 192.30.252.0/22
sudo ufw allow to 20.201.28.152/32
sudo ufw allow to 20.205.243.160/32
sudo ufw allow to 102.133.202.246/32
sudo ufw allow to 20.248.137.50/32
sudo ufw allow to 20.207.73.83/32
sudo ufw allow to 20.27.177.118/32
sudo ufw allow to 20.200.245.248/32
sudo ufw allow to 20.233.54.52/32
sudo ufw allow from 185.199.108.0/22
sudo ufw allow from 140.82.112.0/20
sudo ufw allow from 143.55.64.0/20
sudo ufw allow from 2a0a:a440::/29
sudo ufw allow from 2606:50c0::/32
sudo ufw allow from 192.30.252.0/22
sudo ufw allow from 20.201.28.152/32
sudo ufw allow from 20.205.243.160/32
sudo ufw allow from 102.133.202.246/32
sudo ufw allow from 20.248.137.50/32
sudo ufw allow from 20.207.73.83/32
sudo ufw allow from 20.27.177.118/32
sudo ufw allow from 20.200.245.248/32
sudo ufw allow from 4.16.249.226
sudo ufw allow to 4.16.249.226
sudo ufw allow from 20.233.54.52/32
sudo ufw allow from 127.0.0.1
# VPN ip
sudo ufw allow from $VPNIP
sudo ufw allow to $VPNIP