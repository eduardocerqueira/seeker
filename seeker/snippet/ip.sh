#date: 2026-01-22T17:07:29Z
#url: https://api.github.com/gists/7e80ebbb13f90293374045215c7abc67
#owner: https://api.github.com/users/aleks-mariusz

#!/bin/sh
# ip command wrapper for Debian installer VLAN support
# Ensures parent interface is UP before configuring VLAN interface
# Only triggers on specific commands to avoid breaking other uses

# Check if this is a command that needs VLAN parent interface handling
# We only care about: ip addr add <ip> ... dev <vlan_interface>.<parent_interface>
# and: ip link set up dev <vlan_interface>.<parent_interface>

NEEDS_PARENT_UP=0
VLAN_IFACE=""

# Parse arguments to detect VLAN interface operations
prev_arg=""
for arg in "$@"; do
    case "$prev_arg" in
        dev)
            # This is the interface name after "dev"
            case "$arg" in
                *.*)
                    # Contains a dot - might be VLAN interface
                    VLAN_IFACE="$arg"
                    PARENT_IFACE="${VLAN_IFACE%.*}"
                    # Only treat as VLAN if parent != full name
                    if [ "$PARENT_IFACE" != "$VLAN_IFACE" ]; then
                        NEEDS_PARENT_UP=1
                    fi
                    ;;
            esac
            ;;
    esac
    prev_arg="$arg"
done

# Bring up parent interface if needed
if [ "$NEEDS_PARENT_UP" = "1" ] && [ -n "$VLAN_IFACE" ]; then
    /bin/busybox ip link set up dev "$PARENT_IFACE" 2>/dev/null || true
fi

# Run the real ip command
exec /bin/busybox ip "$@"