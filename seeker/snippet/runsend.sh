#date: 2025-11-25T16:50:19Z
#url: https://api.github.com/gists/3dddcf65d877fdf3bb78f434132a9ea5
#owner: https://api.github.com/users/romshark

#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-}"
DEST_MAC="58:a2:e1:04:a9:db" # top_2 is the destination interface on a connected BlueField-2 NIC I tested against.
COUNT="${2:-1000000}"        # default if not provided
QUEUE=0

if [ -z "$IFACE" ]; then
  echo "Usage: $0 <interface> [packet_count]"
  echo "Example: $0 center_1 5000000"
  exit 1
fi

gcc -O2 -g send.c -o send -lbpf

time sudo ./send \
  -i "$IFACE" \
  -d "$DEST_MAC" \
  -s 192.168.1.10 \
  -D 192.168.1.20 \
  -p 9000 \
  -n "$COUNT" \
  -q $QUEUE \
  -z # Use XDP_ZEROCOPY for high performance on supported hardware. Remove for XDP_COPY mode.