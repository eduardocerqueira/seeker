#date: 2024-09-24T17:03:15Z
#url: https://api.github.com/gists/9c4b8f7aea29fd68b6dee2caadf546ea
#owner: https://api.github.com/users/jacob-MX

#!/bin/bash
 
##################################################################################
# LOAD BALANCE NAT TRAFFIC OVER 2 INTERNET CONNECTIONS WITH DYNAMIC IP ADDRESSES #
##################################################################################

# Traffic that goes over the first connection
# web: 80 443
# email: 25 465 143 993 110 995
# ssh: 22 
PORTS="22 25 80 110 143 443 465 993 995"

# Service providers
ISP1="eth1"
ISP2="eth2"

# Enable IPv4 forwarding
echo "1" > /proc/sys/net/ipv4/ip_forward
 
for port in $PORTS
do
    # Traffic using the specific port will be forwarded to that specifc interface
    # within its desiganted dynamic IP address 
    iptables -t nat -A POSTROUTING -p tcp --dport $port -o $ISP1 -j MASQUERADE
    iptables -t nat -A POSTROUTING -p udp --dport $port -o $ISP1 -j MASQUERADE
    echo "Ipatable load traffic on port $port to $ISP1 interface inserted"
done
 
 
# Traffic NOT matched goes over the 2nd connection
iptables -t nat -A POSTROUTING -o $ISP2 -j MASQUERADE