#date: 2026-01-22T17:07:29Z
#url: https://api.github.com/gists/7e80ebbb13f90293374045215c7abc67
#owner: https://api.github.com/users/aleks-mariusz

#!/bin/sh
# VLAN setup for Debian installer
# Parses vlan= kernel parameter and creates VLAN interface
# Format: vlan=<interface>.<vlan_id>:<parent_interface>
# Example: vlan=eth0.2220:eth0

# Parse kernel command line for vlan= parameter
for param in $(cat /proc/cmdline); do
    case "$param" in
        vlan=*)
            VLAN_SPEC="${param#vlan=}"
            # Extract components from vlan=interface.vlan_id:parent_interface
            VLAN_IFACE="${VLAN_SPEC%:*}"    # e.g., eth0.2220
            PARENT_IFACE="${VLAN_SPEC#*:}"  # e.g., eth0
            VLAN_ID="${VLAN_IFACE#*.}"      # e.g., 2220
            
            # Load 8021q kernel module for VLAN support
            modprobe 8021q 2>/dev/null || true
            
            # Wait for parent interface to appear
            for i in 1 2 3 4 5; do
                if [ -e "/sys/class/net/$PARENT_IFACE" ]; then
                    break
                fi
                sleep 1
            done
            
            # Create VLAN interface
            ip link add link "$PARENT_IFACE" name "$VLAN_IFACE" type vlan id "$VLAN_ID" 2>/dev/null || true
            ;;
    esac
done