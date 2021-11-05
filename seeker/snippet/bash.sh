#date: 2021-11-05T16:54:52Z
#url: https://api.github.com/gists/6ef4593a70bfaaf30710251afce1b4f6
#owner: https://api.github.com/users/Hytreenee

#!/bin/bash

# Init
sudo sed -i "/^# deb .* multiverse$/ s/^# //" /etc/apt/sources.list;
sudo apt update
sudo apt install curl

# Repos
sudo add-apt-repository ppa:atareao/telegram
sudo add-apt-repository ppa:fish-shell/release-3 

# Keys
curl https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo apt-key add

# Postgre
sudo sh -c 'echo "deb https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'

sudo apt update
sudo apt upgrade

# Apps
sudo snap install --classic code
sudo snap install vlc
sudo snap install timeshift

sudo apt install -y bleachbit
sudo apt install -y gparted
sudo apt install -y pavucontrol
sudo apt install -y chrome-gnome-shell
sudo apt install -y gnome-shell-extension-prefs
sudo apt install -y ubuntu-restricted-extras
sudo apt install -y telegram
sudo apt install -y fish
sudo apt install -y preload
sudo apt install -y prelink
sudo apt install -y postgresql-12
sudo apt install -y pgadmin4
sudo apt install -y gnome-tweaks
sudo apt install -y p7zip-full
sudo apt install -y git
sudo apt install -y nautilus-admin
sudo apt install -y flameshot
sudo apt install -y python3-pip
sudo apt install -y nodejs
sudo apt install -y npm
sudo apt install -y fonts-firacode

# get lts nodejs
sudo npm i -g n
sudo n lts

# useful to prevent ssh from dropping connection
sudo sed -i '$a ServerAliveInterval 60' /etc/ssh/ssh_config

#configure pgadmin -> #sudo /usr/pgadmin4/bin/setup-web.sh

# Fix of 2 tg app icons (tg executable located at /opt/telegram)
# WORKS AFTER 1 LAUNCH (since .desktop in .local created after launch)
sudo mv /usr/share/applications/telegram.desktop /usr/share/applications/telegram.!desktop
# copy&change this file to make multiple tdata clients (~/.local/share/Telegram_Desktop)
sudo mv /home/user/.local/share/applications/appimagekit_d98825e589ea79557384fe149efdfbdd-Telegram_Desktop.desktop /home/user/.local/share/applications/Telegram_Desktop_main.desktop

# Disable ubuntu dock
sudo gnome-extensions disable ubuntu-dock@ubuntu.com

# Fix for gnome-shell-extension-prefs
sudo apt-get install gir1.2-gtkclutter-1.0

# Disable ctrl+alt+up/down for vscode compatibility
sudo gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-down "['']"
sudo gsettings set org.gnome.desktop.wm.keybindings switch-to-workspace-up "['']"

# Upgrade
sudo apt update
sudo apt upgrade

# Fish as default shell
# alt solution = usermod -s /usr/bin/fish username
chsh -s `which fish`
#sudo chsh -s `which fish`
usermod -s /usr/bin/fish user
#sudo usermod -s /usr/bin/fish user
# back -> chsh -s (which bash) but no need, type "bash" in fish to get back




# Get OMF (DO MANUALLY, WITH SYSTEM RELOAD BEFORE THAT, FROM FISH, OR SOME BUG HAPPENS!!)
# curl -L https://get.oh-my.fish | fish