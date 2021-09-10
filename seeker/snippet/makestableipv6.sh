#date: 2021-09-10T16:49:49Z
#url: https://api.github.com/gists/5d788583a9b6459311c8b5fd92396706
#owner: https://api.github.com/users/devries

#!/bin/sh

# This file goes in /etc/sysctl.d to set up a stable-privacy IPv6 Address

cat > 50-rfc7217.conf <<EOF
# Set up a secret for RFC7217 private static IPv6 address
net.ipv6.conf.default.stable_secret = $(hexdump -n 16 -e '7/2 "%04x:" 1/2 "%04x" "\n"' /dev/urandom)
EOF
