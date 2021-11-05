#date: 2021-11-05T16:54:28Z
#url: https://api.github.com/gists/e49184304f4a0f180c50a7f0e8481be3
#owner: https://api.github.com/users/sbardu

#!/bin/bash

if [ "$1" = "config" ]; then
        echo "graph_category network"
        echo "graph_title Speedtest"
        echo "graph_args --base 1000 -l 0"
        echo "graph_vlabel DL / UL"
        echo "graph_scale no"
        echo "down.label DL"
        echo "down.type GAUGE"
        echo "down.draw LINE1"
        echo "up.label UL"
        echo "up.type GAUGE"
        echo "up.draw LINE1"
        echo "ping.label PING"
        echo "ping.type GAUGE"
        echo "ping.draw LINE1"
        echo "graph_info Graph of Internet Connection Speed"
        exit 0
fi

OUTPUT=`cat /tmp/speedtest.out`
DOWNLOAD=`echo "$OUTPUT" | grep Download | sed 's/[a-zA-Z:]* \([0-9]*\.[0-9]*\) [a-zA-Z/]*/\1/'`
UPLOAD=`echo "$OUTPUT" | grep Upload | sed 's/[a-zA-Z:]* \([0-9]*\.[0-9]*\) [a-zA-Z/]*/\1/'`
PING=`echo "$OUTPUT" | grep Ping | sed 's/[a-zA-Z:]* \([0-9]*\.[0-9]*\) [a-zA-Z/]*/\1/'`

echo "down.value $DOWNLOAD"
echo "up.value $UPLOAD"
echo "ping.value $PING"
