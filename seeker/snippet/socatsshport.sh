#date: 2025-10-17T16:59:17Z
#url: https://api.github.com/gists/cbdcaae4f96ef6ab89cc18f2c5c4c89b
#owner: https://api.github.com/users/xor2k

#!/bin/sh

if [ $# -ne 3 ]; then
    echo $0 MACHINE SOURCE_PORT_REMOTE TARGET_PORT_LOCAL
    exit
fi

SOCAT=/usr/bin/socat
MACHINE=$1
REMOTE_PORT=$2
LOCAL_PORT=$3

${SOCAT} \
  TCP-LISTEN:${LOCAL_PORT},reuseaddr,fork,keepalive,bind=127.0.0.1,range=127.0.0.1/32 \
  EXEC:"ssh -T -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
              -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
              ${MACHINE} -W 127.0.0.1\:${REMOTE_PORT}"