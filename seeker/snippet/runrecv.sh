#date: 2025-11-25T16:50:19Z
#url: https://api.github.com/gists/3dddcf65d877fdf3bb78f434132a9ea5
#owner: https://api.github.com/users/romshark

#!/usr/bin/env bash
set -euo pipefail

IFACE="${1:-}"

if [ -z "$IFACE" ]; then
  echo "Usage: $0 <interface>"
  exit 1
fi

# Compile the redirector eBPF program
clang -O2 -g -target bpf -c xdp_redirect_kern.c -o xdp_redirect_kern.o

# If something is already attached, detach it (ignore errors)
sudo ip link set dev "$IFACE" xdp off 2>/dev/null || true

# Attach it to the interface
sudo ip link set dev "$IFACE" xdp obj xdp_redirect_kern.o sec xdp

echo "XDP redirector attached to interface: $IFACE"

# Compile receiver
gcc -O2 -g recv.c -o recv -lbpf

echo "Running recv..."

# Run receiver on this interface and UDP port 9000
sudo ./recv "$IFACE" 9000 10
