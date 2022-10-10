#date: 2022-10-10T17:13:58Z
#url: https://api.github.com/gists/95528b099c7feff0c6795391a7bae599
#owner: https://api.github.com/users/Natetronn

#!/bin/bash
# https://wiki.archlinux.org/index.php/Wake-on-LAN

sudo pacman -Syu ethtool

# d (disabled), p (PHY activity), u (unicast activity), m (multicast activity),
# b (broadcast activity), a (ARP activity), and g (magic packet activity)
ethtool <interface> | grep Wake-on
# If not g
ethtool -s <interface> wol g

# Enable persistence using systemd
sudo systemd enable wol@.service

# TODO: "**********"

# Trigger a wake-up, port should be 9 (?)
# Install wol
# TODO: https://www.depicus.com/wake-on-lan/woli

# Find the MAC address
ip link

# In-network
wol target_MAC_address
# or
wol target_internal_IP target_MAC_address

# Across the internet
# Enable port forwarding on target's static IP
wol -p forwarded_port -i router_IP target_MAC_address

# Listen for incoming WOL requests on target computer
sudo pacman -Syu netcat
nc --udp --listen --local-port=9 --hexdump
ump
