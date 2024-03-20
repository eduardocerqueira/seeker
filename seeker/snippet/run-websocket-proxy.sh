#date: 2024-03-20T17:01:30Z
#url: https://api.github.com/gists/00f00809e60ad8005ceffacab5b57658
#owner: https://api.github.com/users/czervenka

#!/bin/sh
# 
# This script requires installed websocket proxy (see readme.md)
# Copy this script to the printer (using scp) and run it with your device key as the only argument

cd /usr/data/websocket-proxy
export KARMEN_URL=https://karmen.fragaria.cz
export NODE_ENV=production
export FORWARD_TO=http://localhost:4408
export SERVER_URL=wss://cloud.karmen.tech
export KEY="$1"
exec node client