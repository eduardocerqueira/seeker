#date: 2025-02-24T16:53:24Z
#url: https://api.github.com/gists/5157593cd8d6b68b74b3aba1df7e219b
#owner: https://api.github.com/users/FoxxMD

#!/usr/bin/bash

LAN=192.168.1.0/16

# delete any existing (prob just RETURN)
iptables -F DOCKER-USER

# accept new ingress from LAN
iptables -A DOCKER-USER -i docker_gwbridge -s $LAN -m state --state NEW -j ACCEPT

# allow egress to LAN if connection is already established
iptables -A DOCKER-USER -i docker_gwbridge -d $LAN -m state --state ESTABLISHED,RELATED -j ACCEPT

# (optional) allow all egress to LAN DNS IP
# remove statement if not needed
iptables -A DOCKER-USER -i docker_gwbridge -d 192.168.1.200 -p udp --dport 53 -j ACCEPT

# drop all others to LAN
iptables -A DOCKER-USER -i docker_gwbridge -d $LAN -j DROP

# move to next chain
iptables -A DOCKER-USER -j RETURN