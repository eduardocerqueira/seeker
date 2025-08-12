#date: 2025-08-12T16:53:29Z
#url: https://api.github.com/gists/faf72d6c2806252151840c322b89e1e7
#owner: https://api.github.com/users/tomg404

#!/bin/sh

# This script is a workaround for a bug that exists
# in Tor since at least 2022. The bug causes the relay to not 
# automatically update its IPv6 address, if the server
# is behind a NAT with dynamic IPv6 prefix.
#
# Related discussion: https://forum.torproject.org/t/ipv6-with-dynamic-prefix-behind-nat/5296
#
# The script should only be used as a parameter of `postscript=`
# in the ddclient.conf file.
#
# The script should have appropriate permissions as it's run
# with sudo.

CURRENT_IP="$1"
LAST_IP_FILE="/var/lib/ddclient/last-ip"
mkdir -p $(dirname $LAST_IP_FILE)

# check if file exists and is not empty
if [ -s "$LAST_IP_FILE" ]; then
        LAST_IP=$(cat "$LAST_IP_FILE")
else
        echo "$CURENT_IP" > "$LAST_IP_FILE"
        exit 1
fi

# update last ip and restart tor if new IP was detected
if [ "$CURRENT_IP" != "$LAST_IP" ]; then
        echo "$CURRENT_IP" > "$LAST_IP_FILE"
        sudo systemctl restart tor@default.service
fi
