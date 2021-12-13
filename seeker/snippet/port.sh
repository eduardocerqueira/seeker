#date: 2021-12-13T16:54:52Z
#url: https://api.github.com/gists/aa3125ad03b05008df9c6617379f0d2c
#owner: https://api.github.com/users/Alcadramin

#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "=> Please run as root.."
  exit
fi

echo "=> Please enter the port you would like to enable ingress.."
read PORT

re='^[0-9]+$'
if ! [[ $PORT =~ $re ]] ; then
   echo "=> Not a valid PORT.." >&2; exit 1
fi

echo "=> Enabling ingress for $PORT.."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport $PORT -j ACCEPT
sudo netfilter-persistent save

echo "=> Job complete.."