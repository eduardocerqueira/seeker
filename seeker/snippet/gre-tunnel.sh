#date: 2022-06-10T16:58:50Z
#url: https://api.github.com/gists/715a39f0c74a774b940e22494e5dca9d
#owner: https://api.github.com/users/Bharat-B

#!/bin/sh

# SERVER A ##
# Make sure IP forwarding is enabled.
echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf
sysctl -p 
# Create a new tunnel via GRE protocol
ip tunnel add tunnel0 mode gre local SERVER_A_IP remote SERVER_B_IP ttl 255
# Add a private subnet to be used on the tunnel
ip addr add 192.168.0.1/30 dev tunnel0
# Turn on the tunnel
ip link set tunnel0 up

## SERVER B ##
# Create a new tunnel via GRE protocol
ip tunnel add tunnel0 mode gre local SERVER_B_IP remote SERVER_A_IP ttl 255
# Add a private subnet to be used on the tunnel
ip addr add 192.168.0.2/30 dev tunnel0
# Turn on the tunnel
ip link set tunnel0 up
# Create a new routing table
echo '100 GRE' >> /etc/iproute2/rt_tables
# Make sure to honor the rules for the private subnet via that table
ip rule add from 192.168.168.0/30 table GRE
# Make sure all traffic goes via SERVER_A ip
ip route add default via 192.168.168.1 table GRE

# Test if the traffic is outgoing through SERVER_A
curl http://ipinfo.io --interface tunnel0

## SERVER A ##
# Accept to and from traffic for the private IP of SERVER_B
iptables -A FORWARD -d 192.168.0.2 -m state --state NEW,ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -s 192.168.0.2 -m state --state NEW,ESTABLISHED,RELATED -j ACCEPT
# Setup a portforward of PORT 80 from SERVER_A to port 8000 in SERVER_B
iptables -t nat -A PREROUTING -d SERVER_A -p tcp -m tcp --dport 80 -j DNAT --to-destination 192.168.0.2:8000