#date: 2024-01-01T17:10:03Z
#url: https://api.github.com/gists/1f3dc4f95557a696fcd5b218b74e645c
#owner: https://api.github.com/users/jagprog5

#!/bin/bash

if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    exit 1
fi

set -e

export NEEDRESTART_MODE=a

apt update && apt upgrade -y

RELEASE=`cat /etc/lsb-release | grep -Po "DISTRIB_CODENAME=\K.*"`

apt install ca-certificates wget net-tools gnupg
wget -qO - https://as-repository.openvpn.net/as-repo-public.gpg | apt-key add -
echo "deb http://as-repository.openvpn.net/as/debian $RELEASE main">/etc/apt/sources.list.d/openvpn-as-repo.list
apt update && apt install openvpn-as -y

URL=`cat /usr/local/openvpn_as/init.log | grep -Po "Admin  UI: \K.*"`
LINE= "**********"
USER= "**********"=\" account with \"[^\"]*\" password.)"`
PASS= "**********"=\" password.)"`

echo "=============================================================="
echo "=============================================================="
echo "=============================================================="
echo "$URL"
echo "username: $USER"
echo "password: "**********"
================="
echo "=============================================================="
echo "=============================================================="
echo "$URL"
echo "username: $USER"
echo "password: $PASS"
