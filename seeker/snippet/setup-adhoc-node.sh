#date: 2023-09-19T17:09:18Z
#url: https://api.github.com/gists/378caebaab2a8056143ae352ee985a33
#owner: https://api.github.com/users/totekuh

#!/bin/bash

# Define network parameters
INTERFACE="$1"
SSID="MyAdHocNetwork"
CHANNEL="1"
IP_ADDR="192.168.1.1"

# Validate the network interface
if [[ -z "$INTERFACE" ]]; then
    echo "Usage: $0 <network_interface>"
    exit 1
fi

# Check if the interface exists
if ! ip link show "$INTERFACE" > /dev/null 2>&1; then
    echo "Error: Network interface $INTERFACE does not exist."
    exit 1
fi

# Disable NetworkManager for the interface
sudo nmcli dev set $INTERFACE managed no

# Set up the network
sudo ip link set $INTERFACE down
sudo iwconfig $INTERFACE mode ad-hoc
sudo iwconfig $INTERFACE essid $SSID
sudo iwconfig $INTERFACE channel $CHANNEL
sudo ip link set $INTERFACE up
sudo ip addr add $IP_ADDR/24 dev $INTERFACE

echo "Ad-Hoc network configured for this node with IP $IP_ADDR."
