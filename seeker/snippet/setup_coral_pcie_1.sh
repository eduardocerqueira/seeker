#date: 2023-01-18T16:48:21Z
#url: https://api.github.com/gists/068364d8d178a32afec712a5ff8a8474
#owner: https://api.github.com/users/maslyankov

#!/bin/bash
echo "Checking if there is a coral present..."
lspci -nn | grep 089a

echo "Checking if there is drivers for it..."
ls /dev/apex_0

echo "Executing install scripts from coral setup tutoral."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gasket-dkms libedgetpu1-std

sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"
sudo groupadd apex
sudo adduser $USER apex
sudo usermod -aG plugdev $USER

read -t 15 -N 1 -p "Rebooting system... Run part two afterwards."
echo
sudo reboot