#date: 2024-06-20T17:04:55Z
#url: https://api.github.com/gists/73c8b13358d7417c4c6f3bfc4af2610f
#owner: https://api.github.com/users/akiross

# Enable the link
ip link set ens18 up

# Set the IP
ip address add 192.168.1.123/24 dev ens18

# Add a route
ip route add default via 192.168.1.1

# Set the nameserver
resolvectl dns ens18 192.168.1.1