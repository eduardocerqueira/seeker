#date: 2022-09-15T17:13:27Z
#url: https://api.github.com/gists/92a78b9ceea0bf7e837d631ccf25c735
#owner: https://api.github.com/users/UnitedWithCode

#!/bin/bash

# updating system
sudo apt update
sudo apt upgrade -y

# installing build tools and python
sudo apt install build-essential python

# installing nodejs
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install nodejs

# install n8n globally
npm install n8n -g

# adding systemd entry
sudo echo "[Unit]
Description=n8n - Easily automate tasks across different services.
After=network.target

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/n8n start --tunnel
Restart=on-failure

[Install]
WantedBy=multi-user.target" > /etc/systemd/system/multi-user.target.wants/n8n.service

# reloading, enabling on boot and starting n8n
sudo systemctl daemon-reload
sudo systemctl enable n8n
sudo systemctl start n8n