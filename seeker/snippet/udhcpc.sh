#date: 2026-01-22T17:07:29Z
#url: https://api.github.com/gists/7e80ebbb13f90293374045215c7abc67
#owner: https://api.github.com/users/aleks-mariusz

#!/bin/sh
# udhcpc wrapper for Debian installer VLAN support
# Ensures parent and VLAN interfaces are UP before running DHCP

# Extract interface name from arguments
IFACE=""
prev_arg=""
for arg in "$@"; do
    case "$prev_arg" in
        -i)
            IFACE="$arg"
            break
            ;;
    esac
    prev_arg="$arg"
done

# Bring up interfaces if needed
if [ -n "$IFACE" ]; then
    # Get parent interface (strip .VLAN if present)
    PARENT="${IFACE%.*}"
    
    # If this is a VLAN interface, bring up parent first
    if [ "$PARENT" != "$IFACE" ]; then
        ip link set up dev "$PARENT" 2>/dev/null || true
    fi
    
    # Bring up the interface
    ip link set up dev "$IFACE" 2>/dev/null || true
fi

# Run the real udhcpc
exec /bin/busybox udhcpc "$@"
