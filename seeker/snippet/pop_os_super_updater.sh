#date: 2022-11-11T17:12:15Z
#url: https://api.github.com/gists/2b7ab1e976ec0c5ab4323b5e78760855
#owner: https://api.github.com/users/kyletimmermans

#!/usr/bin/env bash

echo -e "\nThis program requires that your machine is connected to"
echo -e "a power source and that you are connected to the internet."
echo -e "THIS PROGRAM WILL REBOOT YOUR MACHINE AFTER IT FINISHES!\n"
read -p "Are both of these conditions true? (Y/n) " -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo apt update
    sudo apt upgrade
    sudo apt dist-upgrade
    sudo apt autoremove
    sudo apt autoclean
    sudo fwupdmgr get-devices
    sudo fwupdmgr get-updates
    sudo fwupdmgr update
    flatpak update
    sudo pop-upgrade recovery upgrade from-release
    sudo reboot now
else
    echo "Please connect to a power source and the internet"
    exit 1
fi