#date: 2022-06-03T16:49:17Z
#url: https://api.github.com/gists/30f84ac9c48747b510e6ee39fd9f622e
#owner: https://api.github.com/users/chris24sahadeo

#!/bin/bash

# One liner:
# sudo wget -qO- https://gist.githubusercontent.com/chris24sahadeo/30f84ac9c48747b510e6ee39fd9f622e/raw/55854df431aebfc4520bb3c953c4532cb8d16827/install_vscode.sh | sudo bash 

sudo apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt install apt-transport-https
sudo apt update
sudo apt install code # or code-insiders
