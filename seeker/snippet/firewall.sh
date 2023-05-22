#date: 2023-05-22T16:41:02Z
#url: https://api.github.com/gists/eb44fcd4a58fdc9581f81e8c097cbd33
#owner: https://api.github.com/users/rev-07

#!/bin/bash

export PATH="/usr/sbin:$PATH"

# Set the DDNS hostname
ddns_hostname="CHANGE-HOSTNAME"

# Get the IP address associated with the DDNS hostname
ddns_ip=$(dig +short $ddns_hostname)

# Clear existing firewall rules
ufw --force reset

# Update the UFW rules with the IP address associated with the DDNS hostname
ufw allow from $ddns_ip to any port 22 proto tcp
ufw allow from $ddns_ip to any port 53 proto udp
ufw allow from $ddns_ip to any port 53 proto tcp
ufw allow from $ddns_ip to any port 80 proto tcp
ufw allow from $ddns_ip to any port 443 proto tcp

# Enable the firewall
ufw --force enable