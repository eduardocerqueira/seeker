#date: 2021-11-24T16:54:06Z
#url: https://api.github.com/gists/2bc3c1a4cad98376605ffe371ebb97e6
#owner: https://api.github.com/users/ThePrincelle

iptables -I DOCKER-USER -i <interface_ext> -j DROP
iptables -I DOCKER-USER -m state --state RELATED,ESTABLISHED -j ACCEPT


# Just in case there is a Traefik instance or to add a specific port
iptables -I DOCKER-USER -i <interface_ext> -p tcp --dport 80 -j ACCEPT
iptables -I DOCKER-USER -i <interface_ext> -p tcp --dport 443 -j ACCEPT