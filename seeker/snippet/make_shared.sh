#date: 2022-04-26T17:16:39Z
#url: https://api.github.com/gists/8918f3d16210bd2ea74f0f1073837bf7
#owner: https://api.github.com/users/NetLagina

#!/bin/bash
echo "This script configure shared folder with host. Some actions should be done manually"
read -p "Press any key to start..."
sudo apt-get -y update && sudo apt-get -y dist-upgrade && sudo apt-get -y autoremove
sudo apt-get -y install virtualbox-guest-dkms build-essential linux-headers-virtual
echo "Next step should be done manually"
echo "Please select the Devices from VirtualBox host application menu and click Insert Guest Additions CD image > Run"
read -p "Press any key when VirtualBox Guest Additions software instalation will finished..."
sudo usermod -G vboxsf -a $USER
echo "Please Select the guest machine you want to share with and go to the Settings tab and pick the Shared Folders from the list. From there, click the Plus ( + ) at the top to add a folder you want to share. Then reboot Ubuntu"
read -p "Press any key to exit..."