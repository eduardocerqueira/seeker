#date: 2023-12-26T17:05:01Z
#url: https://api.github.com/gists/61951386314f243b1226d72d486f406b
#owner: https://api.github.com/users/pmdhazy

#/bin/bash

cp attach-usb-devices plug-usb unplug-usb /usr/bin
chmod +x /usr/bin/attach-usb-devices /usr/bin/plug-usb /usr/bin/unplug-usb

cp attach-usb-devices.service /etc/systemd/system
systemctl enable attach-usb-devices

# You can try running systemctl start attach-usb-devices but it'll fail to attach any VMs that are already running.