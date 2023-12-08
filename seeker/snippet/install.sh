#date: 2023-12-08T17:01:33Z
#url: https://api.github.com/gists/1a5610fe5b5351c83b47c898c24f8f27
#owner: https://api.github.com/users/perfecto25

#!/bin/bash

# Get Username
uname=$(whoami)


# Make Folder /opt/rustdesk/
if [ ! -d "/opt/rustdesk" ]; then
    echo "Creating /opt/rustdesk"
    mkdir -p /opt/rustdesk/
    cd /opt/rustdesk/
fi

RDLATEST=$(curl https://api.github.com/repos/rustdesk/rustdesk-server/releases/latest -s | grep "tag_name"| awk '{print substr($2, 2, length($2)-3) }')

wget https://github.com/rustdesk/rustdesk-server/releases/download/${RDLATEST}/rustdesk-server-linux-amd64.zip || { echo "error wget"; exit 1; }
unzip rustdesk-server-linux-x64.zip || { echo "error unzipping"; exit 1; }
rm rustdesk-server-linux-x64.zip

mv amd64/* /opt/rustdesk/

# Setup Systemd to launch hbbs
echo -e "\n[Unit]\nDescription=Rustdesk Signal Server\n\n[Service]\nType=simple\nLimitNOFILE=1000000\nExecStart=/opt/rustdesk/hbbs\nWorkingDirectory=/opt/rustdesk/\nUser=${uname}\nGroup=${uname}\nRestart=always\n# Restart service after 10 seconds if node service crashes\nRestartSec=10\n\n[Install]\nWantedBy=multi-user.target\n" > rdsig.service
sudo cp rdsig.service /etc/systemd/system/rdsig.service
rm rdsig.service

# Setup Systemd to launch hbbr
echo -e "\n[Unit]\nDescription=Rustdesk Relay Server\n\n[Service]\nType=simple\nLimitNOFILE=1000000\nExecStart=/opt/rustdesk/hbbr\nWorkingDirectory=/opt/rustdesk/\nUser=${uname}\nGroup=${uname}\nRestart=always\n# Restart service after 10 seconds if node service crashes\nRestartSec=10\n\n[Install]\nWantedBy=multi-user.target\n" > rdrel.service
sudo cp rdrel.service /etc/systemd/system/rdrel.service
rm rdrel.service

sudo systemctl daemon-reload
sudo systemctl enable rdsig.service
sudo systemctl start rdsig.service
sudo systemctl enable rdrel.service
sudo systemctl start rdrel.service