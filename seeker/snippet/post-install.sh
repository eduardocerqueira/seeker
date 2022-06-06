#date: 2022-06-06T16:59:34Z
#url: https://api.github.com/gists/517509cc3b94d88abf0f192610090c35
#owner: https://api.github.com/users/ale-jr

#!/bin/bash
mkdir -p ~/Downloads/PostInstall
cd ~/Downloads/PostInstall

#install git
sudo apt install git

#install google chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb

#install snap
sudo apt update
sudo apt install snapd -y

#install gnome tweaks
sudo apt install gnome-tweaks -y

#install vs code
sudo snap install --classic code

#install postman
sudo snap install postman

#install node
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.0/install.sh | bash
source ~/.bashrc

nvm install --lts

sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y

sudo usermod -aG docker $USER

# install mockoon
sudo snap install mockoon


#install gitkraken
sudo snap install gitkraken --classic

#install spotify
sudo snap install spotify

#install gnome extensions
sudo apt install gnome-shell-extensions -y

sudo apt install chrome-gnome-shell -y

#install steam

wget https://steamcdn-a.akamaihd.net/client/installer/steam.deb
sudo apt install ./steam.deb
