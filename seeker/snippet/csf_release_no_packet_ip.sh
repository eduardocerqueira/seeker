#date: 2025-01-28T16:51:56Z
#url: https://api.github.com/gists/6e2ef64acd2dde85bc329bf335fe4a98
#owner: https://api.github.com/users/parsibox

#!/bin/bash

# Temporary file to store IPs dropped by iptables in DENYIN chain
TEMP_DROPPED_IPS="/tmp/dropped_ips.txt"

# Function to extract IP blocks/ranges being dropped in the DENYIN chain
get_dropped_ips() {
    # Extract IP blocks/ranges that are actively being dropped in the DENYIN chain
    iptables -vL DENYIN -n | grep 'DROP' | awk '{if ($1 > 0) print $8}' | sort | uniq > "$TEMP_DROPPED_IPS"
}

# Function to check if an IP block/range is dropped in iptables
is_ip_dropped() {
    local ip="$1"
    grep -q "$ip" "$TEMP_DROPPED_IPS"
}

# Function to clean up CSF deny list based on iptables drop status
clean_csf_deny_list() {
    # Loop through each IP in CSF deny list
    while IFS= read -r line; do
        # Skip lines that are comments or don't have an IP (e.g., manual deny)
        if [[ "$line" =~ ^# || -z "$line" ]]; then
            continue
        fi

        # Extract the IP from the line (this works for both IPs and IP ranges)
        blocked_ip=$(echo "$line" | awk '{print $1}')

        # Check if the IP is still being dropped by iptables
        if ! is_ip_dropped "$blocked_ip"; then
            # If the IP is no longer dropped, unblock it using csf -dr
            echo "Unblocking $blocked_ip from CSF deny list."
            csf -dr "$blocked_ip"
        fi
    done < /etc/csf/csf.deny
}

# Main script execution
get_dropped_ips
clean_csf_deny_list

# Reload CSF to apply changes (optional if needed)
csf -r
