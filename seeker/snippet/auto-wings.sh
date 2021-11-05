#date: 2021-11-05T17:12:10Z
#url: https://api.github.com/gists/5175a4c6c4c4614ca499ea6e6c88d598
#owner: https://api.github.com/users/alvesvaren

#!/bin/bash

curl -sSL https://get.docker.com/ | CHANNEL=stable bash
systemctl enable --now docker

mkdir -p /etc/pterodactyl
curl -L -o /usr/local/bin/wings https://github.com/pterodactyl/wings/releases/latest/download/wings_linux_amd64
chmod u+x /usr/local/bin/wings

echo "Enter config command with token:" && read config_command
bash -c "$config_command"

curl https://gist.githubusercontent.com/alvesvaren/5175a4c6c4c4614ca499ea6e6c88d598/raw/wings.service -o /etc/systemd/system/wings.service
systemctl enable --now wings

echo "Done!"
echo
echo "Allocation ip to use:"
hostname -I | awk '{print $1}'