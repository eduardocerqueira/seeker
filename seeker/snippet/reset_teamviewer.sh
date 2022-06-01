#date: 2022-06-01T17:06:27Z
#url: https://api.github.com/gists/5d297cc0bd4c9bbc049e9d827bb86fdc
#owner: https://api.github.com/users/mwikya

#!/bin/bash

##
# Remove "Commercial use suspected"/"Commercial use detected" warning on teamviewer 13
#
# Tested on Arch linux
##

CONFIG_FILE=/opt/teamviewer/config/global.conf

# Make sure only root can run our script
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root" 1>&2
  exit 1
fi

if [ ! -s $CONFIG_FILE  ]; then
  echo "$CONFIG_FILE not found! Teamviewer is installed?" 1>&2
  exit 1
fi

systemctl stop teamviewerd

lastMACUsed=`cat $CONFIG_FILE | grep LastMACUsed | cut -b 23- | tr -d '"'`

for iface in `ls /sys/class/net`; do
  read mac </sys/class/net/$iface/address

  mac=`echo $mac | tr -d ':'`
  if [ "${lastMACUsed#*$mac}" != "$lastMACUsed" ]; then
    echo "$iface -> $mac"
    #ip link set $iface down
    macchanger $iface -r
    #ip link set $iface up
  fi
done

rm -f "$CONFIG_FILE"

systemctl start teamviewerd