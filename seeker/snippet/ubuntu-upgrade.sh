#date: 2022-10-04T17:13:57Z
#url: https://api.github.com/gists/a511f776b39e81f1e6ea5bdef2f7094d
#owner: https://api.github.com/users/willdn

#!/bin/bash
sudo sed -i 's/continue/pass/g' /usr/lib/python3/dist-packages/UpdateManager/Core/MetaRelease.py
sudo sed -i 's/impish/jammy/g' /etc/apt/sources.list
sudo apt-get update
echo "Upgrade distro"
sudo do-release-upgrade
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get install -f -y
sudo apt-get autoremove --purge -y