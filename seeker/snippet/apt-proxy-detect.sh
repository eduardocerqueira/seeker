#date: 2026-03-03T17:26:33Z
#url: https://api.github.com/gists/aaabb412542ed27d458dc79baeba9381
#owner: https://api.github.com/users/thostetler

#!/bin/bash
## Tells  Proxy-Auto-Config if a specified proxy is reachable or if
## it should just fallback to a direct connection.
##
## This file should be executable and referenced by its full path
## in the /etc/apt/apt.conf.d/02proxy file we create too.

ip="192.168.88.1"
port=3142

## This will install netcat automatically if it's missing if uncommented
#if [[ $(which nc >/dev/null; echo $?) -ne 0 ]]; then
#	apt install -y netcat-traditional
#fi

if [[ $(nc -w1 -z $ip $port &>/dev/null; echo $?) -eq 0 ]]; then
    echo -n "http://${ip}:${port}/"
else 
    echo -n "DIRECT"
fi
