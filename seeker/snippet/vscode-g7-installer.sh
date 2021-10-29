#date: 2021-10-29T16:52:02Z
#url: https://api.github.com/gists/d36d0f4d2ee1aacff2a0b6c1f98ec06c
#owner: https://api.github.com/users/alanbixby

#!/usr/bin/env bash
# Created by Alan Bixby (10/29/2021)
# Installs a portable version of Visual Studio code in "$/HOME/.bin", since students lack write access to /usr/bin or sudo access to use the get-apt installer.

# TODO: Tested to be functional, but I still need to add functionality to prevent running on an invalid OS, or from attempting to run it multiple times (after it previously working). 

SAVE_DIR=".bin"

mkdir $HOME/$SAVE_DIR
wget -O "$HOME/$SAVE_DIR/VSCode-linux-x64.tar.gz" "https://code.visualstudio.com/sha/download?build=stable&os=linux-x64"
tar -xf "$HOME/$SAVE_DIR/VSCode-linux-x64.tar.gz" -C $HOME/$SAVE_DIR
rm -r "$HOME/$SAVE_DIR/VSCode-linux-x64.tar.gz"
echo "[Desktop Entry]
Version=1.0
Type=Application
Name=Visual Studio Code
Comment=
Exec=$HOME/$SAVE_DIR/VSCode-linux-x64/bin/code --no-sandbox
Icon=$HOME/$SAVE_DIR/VSCode-linux-x64/resources/app/resources/linux/code.png
Path=
Terminal=false
StartupNotify=true
" >> "$HOME/Desktop/VSCode.desktop"
echo "PATH=\"$PATH:$HOME/$SAVE_DIR\"" >> $HOME/.bashrc
source .bashrc

echo "Visual Studio is installed, and should be visible on your desktop."