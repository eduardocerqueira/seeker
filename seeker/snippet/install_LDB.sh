#date: 2025-08-14T17:02:11Z
#url: https://api.github.com/gists/97135365fbdd10b0e83dd31d0debb5da
#owner: https://api.github.com/users/Manuelmik

#!/bin/bash

# Exit on error
set -e

# Update and install required packages
sudo apt update
sudo apt install cifs-utils psmisc python3 git python3-serial -y

# Mount samba share and clone repository
sudo mkdir -p /mnt/smb
sudo mount -t cifs //mwnas1/MWGit/ /mnt/smb/ -o vers=2.0,guest
sudo mkdir -p /opt/LaserDriveBoxSoftware
cd /opt/LaserDriveBoxSoftware
sudo git clone /mnt/smb/LDriveBox_CM5.git
sudo umount /mnt/smb

# Create log directory
cd /var/log
sudo mkdir -p LaserDriveBox
sudo chown -R mw:mw LaserDriveBox

# Configure UART
sudo raspi-config nonint do_serial_hw 0

# Create and configure systemd service
cat << EOF | sudo tee /etc/systemd/system/LaserDriveBox.service
[Unit]
Description=LaserDriveBox main Process
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=3
ExecStart=/usr/bin/python /opt/LaserDriveBoxSoftware/LDriveBox_CM5/Python_Reimplementation/main.py
WorkingDirectory=/opt/LaserDriveBoxSoftware/LDriveBox_CM5/Python_Reimplementation
User=mw
Group=mw

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable LaserDriveBox.service

# Configure ethernet
cat << EOF | sudo tee /etc/NetworkManager/system-connections/ethernet.nmconnection
[connection]
type=ethernet
id=my_eth
autoconnect-priority=-999
interface-name=eth0
timestamp=1755015631

[ipv4]
address1=192.168.0.220/24,192.168.0.1
method=manual

[ipv6]
addr-gen-mode=default
method=auto
EOF

sudo chmod 600 /etc/NetworkManager/system-connections/ethernet.nmconnection
sudo nmcli connection reload
sudo nmcli connection up my_eth

# Reboot system
sudo reboot now