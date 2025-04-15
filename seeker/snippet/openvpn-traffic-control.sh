#date: 2025-04-15T17:03:41Z
#url: https://api.github.com/gists/11eb850a37e832c293999d2f84a6133a
#owner: https://api.github.com/users/m00nbyte

#!/bin/bash

# ======================================================================================
#
# © 2025 m00nbyte
#
# Script Name:       openvpn-traffic-control.sh
# Description:       Manage bandwidth limits and traffic quotas for OpenVPN clients.
# Author:            m00nbyte
# Date Created:      2024-11-11
# Last Modified:     2025-04-15
# Version:           1.0.0
# License:           MIT License
#
#
# Features:
# --------------------------------------------------------------------------------------
#    - Setup OpenVPN and subscriptions
#    - Manage users and subscriptions
#    - Set bandwidth limits
#    - Set traffic quota limits
#    - IP allocation and reuse
#    - Client isolation
#    - No server logs
#    - AdGuard DNS server
#    - Restrict SSH access to IP
#
#
# Tested On:
# --------------------------------------------------------------------------------------
#    - Ubuntu 24.04
#    - OpenVPN 2.6.12
#    - iptables 1.8.10
#
#
# Usage:
# --------------------------------------------------------------------------------------
#
# 1. Interactive Mode (Text-based menu)
#    Run the script to navigate through the menu and select options:
#
#    $ ./openvpn-traffic-control.sh
#
#
# 2. Command-Line Interface Mode (CLI)
#    Directly execute specific commands with options:
#
#    $ ./openvpn-traffic-control.sh <command> [options]
#
#    Command Examples:
#       --setup <hostname> <cert_name>                    : Initial setup
#       --ssh <client_ip>                                 : Restrict SSH access to IP
#       --quota view                                      : View current traffic quotas
#       --quota update                                    : Update traffic quotas
#       --client add <cert_name> <subscription>           : Add new client
#       --client update <cert_name> <subscription>        : Update client subscription
#       --client remove <cert_name>                       : Remove client from system
#
# ======================================================================================

# --------------------------------------------------------------------------------------
# Global Configuration
# --------------------------------------------------------------------------------------

# Network
NETWORK_INTERFACE="tun0"                # Network interface used by OpenVPN
CIDR_IPV4="10.8.0.0/24"                 # IPv4 CIDR block used by OpenVPN
CIDR_IPV6="fddd:1194:1194:1194::100/64" # IPv6 CIDR block used by OpenVPN
BASE_IPV4=${CIDR_IPV4%.*.*}             # First three IPv4 octets
BASE_IPV6=${CIDR_IPV6%%/*}              # First three IPv6 sections

# Script
SCRIPT_FULL_PATH="$(realpath "$0")"                     # Full script path
SCRIPT_BASE_DIR="$(dirname "$SCRIPT_FULL_PATH")"        # Current script directory
SCRIPT_DATA_DIR="$SCRIPT_BASE_DIR/otc"                  # Script data directory
OPENVPN_SETUP_FILE="$SCRIPT_DATA_DIR/openvpn-setup.sh"  # OpenVPN setup file
SUBSCRIPTIONS_FILE="$SCRIPT_DATA_DIR/subscriptions.txt" # Subscriptions file
TRAFFIC_LOG="$SCRIPT_DATA_DIR/traffic_usage.txt"        # Traffic usage log
QUOTA_HISTORY="$SCRIPT_DATA_DIR/traffic_history.txt"    # Traffic history log
LAST_QUOTA_RESET="$SCRIPT_DATA_DIR/last_quota_reset"    # Last quota reset

# OpenVPN
OPENVPN_BASE_DIR="/etc/openvpn"                               # OpenVPN base directory
OPENVPN_SERVICE_FILE="/lib/systemd/system/openvpn@.service"   # OpenVPN systemd service file
OPENVPN_IPP_FILE="$OPENVPN_CONFIG_DIR/ipp.txt"                # OpenVPN IP address mapping file
OPENVPN_CONFIG_DIR="$OPENVPN_BASE_DIR/server"                 # OpenVPN server config directory
OPENVPN_CCD_DIR="$OPENVPN_BASE_DIR/ccd"                       # OpenVPN client config directory
OPENVPN_SERVER_CONFIG="$OPENVPN_CONFIG_DIR/server.conf"       # OpenVPN server config file
OPENVPN_CLIENT_CONFIG="$OPENVPN_CONFIG_DIR/client-common.txt" # OpenVPN client config file

# Subscription limits for different plans
#
# Format: ["plan_name"]="download_speed upload_speed monthly_quota"
#
# download_speed - maximum download speed in Mbit (Megabits per second)
# upload_speed   - maximum upload speed in Mbit (Megabits per second)
# monthly_quota  - monthly data limit in GB (Gigabytes)
declare -A SUBSCRIPTION_LIMITS=(
    ["bronze"]="10 10 50"
    ["silver"]="50 50 100"
    ["gold"]="100 100 500"
    ["unlimited"]="1000 1000 0"
)

# List of subscription plans in the desired order
SUBSCRIPTION_LIST=("bronze" "silver" "gold" "unlimited")

# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------

# Log a message to the console
log_message() {
    local message="$1"

    local timestamp
    timestamp="$(date +"%d.%m.%Y %H:%M:%S")"

    echo -e "$timestamp | $message"
}

# Wait for a keypress
wait_for_keypress() {
    echo -e "\nPress any key to return to the menu..."
    read -r -n 1 -s
}

# Converts a byte size to a human-readable format
format_size() {
    local bytes=$1

    # If size is less than 1 KB, display in bytes
    if [[ "$bytes" -lt 1024 ]]; then
        echo "${bytes} bytes"

    # If size is less than 1 MB, display in KB
    elif [[ "$bytes" -lt 1048576 ]]; then
        echo "$(echo "scale=2; $bytes/1024" | bc) KB"

    # If size is less than 1 GB, display in MB
    elif [[ "$bytes" -lt 1073741824 ]]; then
        echo "$(echo "scale=2; $bytes/1048576" | bc) MB"

    # For sizes 1 GB and above, display in GB
    else
        echo "$(echo "scale=2; $bytes/1073741824" | bc) GB"
    fi
}

# Calculate percentage of used quota
calculate_percentage() {
    local value_in_bytes=$1
    local total_in_gb=$2

    # Convert bytes to MB
    local value_in_mb=$((value_in_bytes / 1024 / 1024))

    # Convert GB to MB
    local total_in_mb=$((total_in_gb * 1024))

    # Calculate percentage and trim to two decimal places
    local percentage
    percentage=$(echo "scale=4; ($value_in_mb / $total_in_mb) * 100" | bc -l)
    percentage=${percentage%??}

    # Output the result
    echo "${percentage}%"
}

# Restarts the OpenVPN service and the tun0 network interface
restart_services() {
    log_message "Restarting services..."

    sudo systemctl restart openvpn-server@server.service # Restart the OpenVPN service
    sudo ip link set tun0 down                           # Bring down the network interface (tun0)
    sudo ip link set tun0 up                             # Bring up the network interface (tun0)

    log_message "Services restarted successfully."
}

# Adds a cron job to update quotas at a specified interval
setup_cron() {
    log_message "Adding quota update cron job..."

    # Define the cron job
    local cron_job="*/1 * * * * $SCRIPT_FULL_PATH --quota update"

    # Check if the cron job does not exist
    if ! crontab -l 2>/dev/null | grep -qF "$cron_job"; then
        (
            crontab -l 2>/dev/null # List existing cron jobs
            echo "$cron_job"       # Add the new cron job command
        ) | crontab -              # Update crontab with the new job list

        log_message "Quota update cron job added."
    else
        log_message "Quota update cron job already exists."
    fi
}

# Restrict SSH access
restrict_ssh_access() {
    local new_ip="$1"

    # Delete all rules in the INPUT chain matching SSH (dpt:ssh)
    while iptables -L INPUT --line-numbers -v | grep 'dpt:ssh'; do
        # Always delete the first matching rule for dpt:ssh
        iptables -D INPUT "$(iptables -L INPUT --line-numbers -v | grep 'dpt:ssh' | awk 'NR==1 {print $1}')"
    done

    # Remove any existing DROP rule for SSH (this should ideally be the last rule)
    iptables -D INPUT -p tcp --dport 22 -j DROP 2>/dev/null

    # Check if the new IP already has an ACCEPT rule, if not, add it
    if ! iptables -C INPUT -s "$new_ip" -p tcp --dport 22 -j ACCEPT 2>/dev/null; then
        iptables -A INPUT -s "$new_ip" -p tcp --dport 22 -j ACCEPT
    fi

    # Add the DROP rule for SSH from all other IPs
    iptables -A INPUT -p tcp --dport 22 -j DROP
}

# --------------------------------------------------------------------------------------
# OpenVPN
# --------------------------------------------------------------------------------------

# Install and setup OpenVPN with predefined configuration
setup_openvpn_server() {
    local hostname="$1"
    local cert_name="$2"

    log_message "Setup OpenVPN server..."

    # Check if the setup file already exists
    if [ -f "$OPENVPN_SETUP_FILE" ]; then
        log_message "Setup already exists. Skipping download."
    else
        # Download the auto-setup script
        wget -O "$OPENVPN_SETUP_FILE" https://get.vpnsetup.net/ovpn
    fi

    # Check if the OpenVPN is already installed
    if command -v openvpn &>/dev/null; then
        log_message "OpenVPN is already installed. Skipping installation."
    else
        # Install and configure OpenVPN
        sudo bash "$OPENVPN_SETUP_FILE" --auto \
            --serveraddr "$hostname" \
            --proto UDP \
            --port 1194 \
            --clientname "$cert_name" \
            --dns1 94.140.14.14 \
            --dns2 94.140.15.15
    fi

    log_message "OpenVPN setup complete."
}

# Patch the OpenVPN configuration files
patch_openvpn_config() {
    local label="# Patched"

    log_message "Patching OpenVPN configuration..."

    # Check if the config is already patched
    if grep -q "$label" "$OPENVPN_SERVER_CONFIG"; then
        log_message "OpenVPN configuration is already patched."
        return 0
    fi

    sed -i 's/^verb 3$/verb 2/' "$OPENVPN_CLIENT_CONFIG"                # Reducde the verbosity in client logs
    sed -i 's/^verb 3$/verb 2/' "$OPENVPN_SERVER_CONFIG"                # Reducde the verbosity in server logs
    sed -i '/^ifconfig-pool-persist ipp.txt/d' "$OPENVPN_SERVER_CONFIG" # Remove the existing ip pool option

    echo "" >>"$OPENVPN_SERVER_CONFIG"                  # Append a empty line before making changes
    sed -i '/^[[:space:]]*$/d' "$OPENVPN_SERVER_CONFIG" # Remove all empty lines at the end of the file

    # Define options to be added
    local lines=(
        "$label"
        "ifconfig-pool-persist $OPENVPN_IPP_FILE 0"
        "client-config-dir $OPENVPN_CCD_DIR"
        "ccd-exclusive"
        # Only for debugging
        # "log /var/log/openvpn.log"
    )

    # Prepend a empty line before adding the new options
    echo "" >>"$OPENVPN_SERVER_CONFIG"

    # Append the new options at the end
    for line in "${lines[@]}"; do
        echo "$line" >>"$OPENVPN_SERVER_CONFIG"
    done

    log_message "OpenVPN configuration patched."
}

# Patch the OpenVPN systemd service file
patch_openvpn_service() {
    log_message "Patching OpenVPN service..."

    # Define the correct options
    local working_directory="WorkingDirectory=$OPENVPN_CONFIG_DIR"
    local exec_start="ExecStart=/usr/sbin/openvpn --daemon ovpn-%i --status /run/openvpn/%i.status 10 --cd $OPENVPN_CONFIG_DIR --script-security 2 --config $OPENVPN_CONFIG_DIR/%i.conf --writepid /run/openvpn/%i.pid"

    # Define an array of search patterns and replacement strings
    declare -A patterns=(
        ["^WorkingDirectory=.*$"]="$working_directory"
        ["^ExecStart=/usr/sbin/openvpn.*$"]="$exec_start"
    )

    # Track changes
    local changed=false

    # Loop over the patterns array
    for pattern in "${!patterns[@]}"; do
        if grep -qE "$pattern" "$OPENVPN_SERVICE_FILE"; then
            # Check if the existing line matches the desired content
            local current_value
            current_value=$(grep -E "$pattern" "$OPENVPN_SERVICE_FILE")

            if [ "$current_value" != "${patterns[$pattern]}" ]; then
                # Replace the entire line that matches the pattern
                sed -i -E "s|$pattern|${patterns[$pattern]}|" "$OPENVPN_SERVICE_FILE"
                changed=true
            fi
        fi
    done

    # Reload systemd to apply the updated configuration
    if [ "$changed" = true ]; then
        systemctl daemon-reload
    else
        log_message "OpenVPN service is already patched."
    fi

    log_message "OpenVPN service patched."
}

# --------------------------------------------------------------------------------------
# IP Address Mapping
# --------------------------------------------------------------------------------------

# Updates client mappings by assigning IP addresses and creating CCD files for each client
update_client_mapping() {
    log_message "Updating client mapping..."

    local ipv4_counter=2 # IP v4 counters
    local ipv6_counter=0 # IP v6 counters

    # Reset the OPENVPN_IPP_FILE
    : >"$OPENVPN_IPP_FILE"

    # Read each line from SUBSCRIPTIONS_FILE
    while IFS='=' read -r cert_name subscription; do
        # Skip any line that doesn't match the format
        [[ -z $cert_name || -z $subscription ]] && continue

        local new_ipv4="${BASE_IPV4}${ipv4_counter}"     # New IP v4 address
        local new_ipv6="${BASE_IPV6}${ipv6_counter}"     # New IP v6 address
        local ccd_file="${OPENVPN_CCD_DIR}/${cert_name}" # CCD file path

        # Write client mapping into OPENVPN_IPP_FILE
        echo "${cert_name},${new_ipv4},${new_ipv6}" >>"$OPENVPN_IPP_FILE"

        # Create ccd file
        if [[ ! -f "$ccd_file" ]]; then
            log_message "Creating ccd file for $cert_name"

            touch "$ccd_file"                                           # Create the CCD file for the client
            echo "ifconfig-push ${new_ipv4} 255.255.255.0" >"$ccd_file" # Add the assigned IPv4 address
            echo "ifconfig-ipv6-push ${new_ipv6}/64" >>"$ccd_file"      # Add the assigned IPv6 address
            chmod 0644 "$ccd_file"                                      # Set file permissions to ensure proper access

            log_message "Created ccd file for $cert_name | $new_ipv4 | $new_ipv6"
        fi

        # Increment the IP address counter
        ((ipv4_counter++))
        ((ipv6_counter++))
    done <"$SUBSCRIPTIONS_FILE"

    log_message "Client mapping updated."
}

# --------------------------------------------------------------------------------------
# Subscriptions
# --------------------------------------------------------------------------------------

# Setup IP subscriptions
setup_subscriptions() {
    log_message "Creating traffic subscription chain..."

    # Block communication between clients (must be the very first FORWARD rule)
    iptables -C FORWARD -s "$CIDR_IPV4" -d "$CIDR_IPV4" -j DROP 2>/dev/null || iptables -I FORWARD -s "$CIDR_IPV4" -d "$CIDR_IPV4" -j DROP

    # Check if the SUBSCRIPTION chain exists, and create it if it doesn't
    iptables -L SUBSCRIPTION -n >/dev/null 2>&1 || iptables -N SUBSCRIPTION

    # Clear all existing entries
    iptables -F SUBSCRIPTION

    # Ensure that traffic on INPUT, OUTPUT, and FORWARD chains passes through the SUBSCRIPTION chain
    iptables -C INPUT -j SUBSCRIPTION 2>/dev/null || iptables -I INPUT -j SUBSCRIPTION
    iptables -C OUTPUT -j SUBSCRIPTION 2>/dev/null || iptables -I OUTPUT -j SUBSCRIPTION

    # Use position `2` to place the rule directly after DROP
    iptables -C FORWARD -j SUBSCRIPTION 2>/dev/null || iptables -I FORWARD 2 -j SUBSCRIPTION

    log_message "Traffic subscription chain created."
}

# Update IP subscriptions
update_subscriptions() {
    log_message "Updating traffic subscription rules..."

    # Get all IPs from the file
    declare -A file_ips

    # Read each line from OPENVPN_IPP_FILE
    while IFS=, read -r cert_name ip_v4 _; do
        # Skip any line that doesn't match the format
        [ -z "$cert_name" ] || [ -z "$ip_v4" ] && continue

        file_ips["$ip_v4"]=1
    done <"$OPENVPN_IPP_FILE"

    # Get all IPs currently in the iptables SUBSCRIPTION chain
    local current_ips
    current_ips=$(iptables -L SUBSCRIPTION -n | awk '/RETURN/ {print $4}')

    # Process current iptables rules
    for ip in $current_ips; do
        # Skip 0.0.0.0/0
        if [[ "$ip" == "0.0.0.0/0" || -z "$ip" ]]; then
            continue
        fi

        if [[ -z "${file_ips["$ip"]}" ]]; then
            # If the IP is not in the file, remove its rules
            iptables -D SUBSCRIPTION -s "$ip" -i "$NETWORK_INTERFACE" -j RETURN 2>/dev/null
            iptables -D SUBSCRIPTION -d "$ip" -o "$NETWORK_INTERFACE" -j RETURN 2>/dev/null

            log_message "Removed iptables rules for $ip"
        fi
    done

    # Process file entries
    for ip in "${!file_ips[@]}"; do
        # Check if there is already a rule for this IP
        if ! iptables -L SUBSCRIPTION -n | grep -q "$ip"; then
            # Add rules if missing
            iptables -A SUBSCRIPTION -s "$ip" -i "$NETWORK_INTERFACE" -j RETURN
            iptables -A SUBSCRIPTION -d "$ip" -o "$NETWORK_INTERFACE" -j RETURN

            log_message "Added iptables rules for $ip"
        fi
    done

    log_message "Traffic subscription rules updated."
}

# --------------------------------------------------------------------------------------
# Bandwidth Limits
# --------------------------------------------------------------------------------------

# Setup bandwidth limits based on subscription levels
setup_bandwidth_limits() {
    log_message "Applying bandwidth limits..."

    tc qdisc del dev "$NETWORK_INTERFACE" root 2>/dev/null                      # Remove root queueing discipline
    tc qdisc del dev "$NETWORK_INTERFACE" ingress 2>/dev/null                   # Remove ingress queueing discipline
    tc qdisc add dev "$NETWORK_INTERFACE" root handle 1: htb default 30 r2q 100 # Add htb for traffic shaping on root
    tc qdisc add dev "$NETWORK_INTERFACE" handle ffff: ingress                  # Add ingress qdisc to handle incoming traffic

    # Store the last IP address
    local last_ip=""

    # Read each line from OPENVPN_IPP_FILE
    while IFS=, read -r cert_name ip_v4 _; do
        # Skip any line that doesn't match the format
        [ -z "$cert_name" ] || [ -z "$ip_v4" ] && continue

        # Update the last IP address
        last_ip="$ip_v4"

        # Get the subscription level of the client
        local subscription
        subscription=$(grep -i "^$cert_name=" "$SUBSCRIPTIONS_FILE" | cut -d'=' -f2 | tr -d '[:space:]')

        # Skip if no subscription level found
        [[ -z "$subscription" ]] && continue

        # Get bandwidth limits for the subscription level
        local limits=("${SUBSCRIPTION_LIMITS["$subscription"]}")
        local download_limit="${limits[0]}mbit"
        local upload_limit="${limits[1]}mbit"

        # Create a unique class ID for each client based on the IP address
        local class_id
        class_id="1:$(echo "$ip_v4" | awk -F. '{print $4}')"

        # Add the class for the download limit
        tc class add dev "$NETWORK_INTERFACE" parent 1: classid "$class_id" htb rate "${download_limit}"

        # Apply download limit for destination IP address
        tc filter add dev "$NETWORK_INTERFACE" protocol ip parent 1: prio 1 u32 match ip dst "$ip_v4" flowid "$class_id"

        # Apply upload limit for source IP address
        tc filter add dev "$NETWORK_INTERFACE" parent ffff: protocol ip u32 match ip src "$ip_v4" police rate "${upload_limit}" burst 200k drop flowid :1

        log_message "Bandwidth limits applied for $cert_name | $ip_v4"
    done <"$OPENVPN_IPP_FILE"

    # Apply a global limit to the rest
    if [ -n "$last_ip" ]; then
        # Extract the last octet of the last assigned IP address
        local last_ip_last_octet
        last_ip_last_octet=$(echo "$last_ip" | awk -F. '{print $4}')

        # Set the range of IPs to limit
        local start_ip=$((last_ip_last_octet + 1))
        local end_ip=254

        # Apply the limits for the range of IPs
        for unused_octet in $(seq $start_ip $end_ip); do
            local unused_ip_v4="$BASE_IPV4.$unused_octet"

            # Add the class for the download limit
            tc class add dev "$NETWORK_INTERFACE" parent 1: classid "1:$unused_octet" htb rate 1mbit ceil 1mbit

            # Apply download limit for destination IP address
            tc filter add dev "$NETWORK_INTERFACE" protocol ip parent 1: prio 1 u32 match ip dst "$unused_ip_v4" flowid "1:$unused_octet"

            # Apply upload limit for source IP address
            tc filter add dev "$NETWORK_INTERFACE" parent ffff: protocol ip u32 match ip src "$unused_ip_v4" police rate 1mbit burst 200k drop flowid :1
        done

        log_message "Bandwidth limtis applied for $BASE_IPV4.$start_ip to $BASE_IPV4.$end_ip"
    fi

    log_message "Bandwidth limits applied."
}

# --------------------------------------------------------------------------------------
# Traffic Usage
# --------------------------------------------------------------------------------------

# Log current traffic usage for each client
log_traffic_usage() {
    # Reset the TRAFFIC_LOG
    : >"$TRAFFIC_LOG"

    # Read each line from OPENVPN_IPP_FILE
    while IFS=, read -r cert_name ip_v4 _; do
        # Skip any line that doesn't match the format
        [ -z "$cert_name" ] || [ -z "$ip_v4" ] && continue

        # Retrieve transmitted bytes for the IP address
        local tx_bytes
        tx_bytes=$(iptables -L SUBSCRIPTION -v -x | grep "$ip_v4" | awk '{print $2}' | head -1)

        # Retrieve received bytes for the IP address
        local rx_bytes
        rx_bytes=$(iptables -L SUBSCRIPTION -v -x | grep "$ip_v4" | awk '{print $2}' | tail -1)

        # Fallback if no traffic data was found
        tx_bytes=${tx_bytes:-0}
        rx_bytes=${rx_bytes:-0}

        # Log the traffic usage for each client
        echo "$cert_name,$ip_v4,TX:${tx_bytes} bytes,RX:${rx_bytes} bytes" >>"$TRAFFIC_LOG"
    done <"$OPENVPN_IPP_FILE"
}

# --------------------------------------------------------------------------------------
# Archive Traffic Log
# --------------------------------------------------------------------------------------

# Archive traffic log before clearing it
archive_traffic_log() {
    log_message "Archiving traffic log for $last_reset_month..."

    # Add the log contents
    {
        echo "# $last_reset_month"
        cat "$TRAFFIC_LOG"
    } >>"$QUOTA_HISTORY"

    log_message "Traffic log archived."
}

# --------------------------------------------------------------------------------------
# Reset Traffic Usage
# --------------------------------------------------------------------------------------

# Check and reset traffic quotas at the start of a new month
reset_log_quotas() {
    log_message "Checking quota reset..."

    # Get the current month
    local current_month
    current_month=$(date +%Y-%m)

    # Get last reset date, or default to "never"
    local last_reset_month
    last_reset_month=$([ -f "$LAST_QUOTA_RESET" ] && cat "$LAST_QUOTA_RESET" || echo "never")

    log_message "Last quota reset: $last_reset_month"

    # Compare the stored last reset month with the current month
    if [ "$last_reset_month" != "$current_month" ]; then
        if [ "$last_reset_month" != "never" ]; then
            archive_traffic_log "$last_reset_month"
        fi

        log_message "Resetting traffic usage for $current_month..."

        # Reset the traffic log and counters
        : >"$TRAFFIC_LOG"
        iptables -Z SUBSCRIPTION
        echo "$current_month" >"$LAST_QUOTA_RESET"

        log_message "Traffic usage reset for $current_month."
    fi
}

# --------------------------------------------------------------------------------------
# Update Traffic Usage
# --------------------------------------------------------------------------------------

# Apply bandwidth limits and quotas based on the subscription level or just display them
handle_limits_quotas() {
    if [[ "$action" == "update" ]]; then
        log_message "Updating bandwidth limits and quotas..."
    fi

    local action="$1"

    # Read each line from OPENVPN_IPP_FILE
    while IFS=, read -r cert_name ip_v4 _; do
        # Skip any line that doesn't match the format
        [ -z "$cert_name" ] || [ -z "$ip_v4" ] && continue

        # Get the subscription level of the client
        local subscription
        subscription=$(grep -i "^$cert_name=" "$SUBSCRIPTIONS_FILE" | cut -d'=' -f2 | tr -d '[:space:]')

        # Get quota limit for the subscription level
        local limits=("${SUBSCRIPTION_LIMITS["$subscription"]}")
        local quota_gb="${limits[2]}"

        # Convert quota to bytes
        local quota_bytes=$((quota_gb * 1024 * 1024 * 1024))

        # Fetch usage information
        local usage_info
        usage_info=$(grep "$ip_v4" "$TRAFFIC_LOG")

        # Extract TX values
        local used_tx
        used_tx=$(echo "$usage_info" | grep -o "TX:[^,]*" | cut -d':' -f2 | awk '{print $1}')

        # Extract RX values
        local used_rx
        used_rx=$(echo "$usage_info" | grep -o "RX:[^,]*" | cut -d':' -f2 | awk '{print $1}')

        # Fallback if no traffic data was found
        used_tx=${used_tx:-0}
        used_rx=${used_rx:-0}

        # Set total traffic usage for the IP address
        local total_used_bytes=$((used_tx + used_rx))

        # Display usage for `unlimited` subscription
        if [[ "$subscription" == "unlimited" ]]; then
            log_message "$cert_name | $subscription | $ip_v4 | $(format_size "$total_used_bytes") of ∞ GB"

            if [[ "$action" == "update" ]]; then
                tc qdisc del dev "$NETWORK_INTERFACE" root 2>/dev/null    # Remove root queueing discipline
                tc qdisc del dev "$NETWORK_INTERFACE" ingress 2>/dev/null # Remove ingress queueing discipline

                # Remove all traffic control rules related to this IP address
                tc filter del dev "$NETWORK_INTERFACE" protocol ip parent 1: prio 1 u32 match ip dst "$ip_v4" flowid 1:10 2>/dev/null
                tc filter del dev "$NETWORK_INTERFACE" parent ffff: protocol ip u32 match ip src "$ip_v4" flowid :1 2>/dev/null
            fi

            continue
        fi

        # If quota is exceeded or subscription is invalid, apply a strict bandwidth limit
        if ((total_used_bytes >= quota_bytes)); then
            log_message "\033[1;31m$cert_name | $subscription | $ip_v4 | $(format_size "$total_used_bytes") of ${quota_gb} GB | $(calculate_percentage "$total_used_bytes $quota_gb")\033[0m"

            if [[ "$action" == "update" ]]; then
                # Create a unique class ID for each client based on the IP address
                local class_id
                class_id="1:$(echo "$ip_v4" | awk -F. '{print $4}')"

                # Update ingress limit on the traffic class
                tc class change dev "$NETWORK_INTERFACE" classid "$class_id" htb rate 1mbit ceil 1mbit

                # Update egress limit by matching internal source IP with the policing rate
                tc filter change dev "$NETWORK_INTERFACE" parent ffff: protocol ip u32 match ip src "$ip_v4" police rate 1mbit burst 200k drop flowid :1
            fi
        else
            # Quota is within limits
            log_message "$cert_name | $subscription | $ip_v4 | $(format_size "$total_used_bytes") of ${quota_gb} GB | $(calculate_percentage "$total_used_bytes $quota_gb")"
        fi
    done <"$OPENVPN_IPP_FILE"
}

# --------------------------------------------------------------------------------------
# Client Management
# --------------------------------------------------------------------------------------

# Manage clients and subscriptions
manage_client() {
    local action="$1"
    local cert_name="$2"
    local subscription="$3"

    # List clients
    if [[ "$action" == "list" ]]; then
        # Read each line from OPENVPN_IPP_FILE
        while IFS=, read -r cert_name ip_v4 _; do
            # Skip any line that doesn't match the format
            [ -z "$cert_name" ] || [ -z "$ip_v4" ] && continue

            echo "$cert_name | $ip_v4"
        done <"$OPENVPN_IPP_FILE"

    # Add new client
    elif [[ "$action" == "add" ]]; then
        # Check if the client already exists in the SUBSCRIPTIONS_FILE
        if grep -qi "^$cert_name=" "$SUBSCRIPTIONS_FILE"; then
            log_message "Error: Client $cert_name already exists in subscriptions file."
            return 1
        fi

        # Generate a client certificate and add the client to OpenVPN
        sudo bash "$OPENVPN_SETUP_FILE" --addclient "$cert_name"

        # Add the client to the SUBSCRIPTIONS_FILE
        echo "$cert_name=$subscription" >>"$SUBSCRIPTIONS_FILE"
        log_message "Client $cert_name added with subscription '$subscription'."

    # Update client subscription
    elif [[ "$action" == "update" ]]; then
        # Check if the client exists in the SUBSCRIPTIONS_FILE
        if ! grep -qi "^$cert_name=" "$SUBSCRIPTIONS_FILE"; then
            log_message "Error: Client $cert_name not found in subscriptions file."
            return 1
        fi

        # Update the subscription type in the SUBSCRIPTIONS_FILE
        sed -i "s/^$cert_name=[^=]*/$cert_name=$subscription/" "$SUBSCRIPTIONS_FILE"

        log_message "Subscription for client $cert_name updated to '$subscription'."

    # Remove client
    elif [[ "$action" == "remove" ]]; then
        # Check if the client exists in the SUBSCRIPTIONS_FILE
        if ! grep -qi "^$cert_name=" "$SUBSCRIPTIONS_FILE"; then
            log_message "Error: Client $cert_name not found in subscriptions file."
            return 1
        fi

        # Revoke the client certificate and remove the client in OpenVPN
        sudo bash "$OPENVPN_SETUP_FILE" -y --revokeclient "$cert_name"

        # Remove the client from the SUBSCRIPTIONS_FILE
        sed -i "/^$cert_name=/d" "$SUBSCRIPTIONS_FILE"
        sed -i '/^$/d' "$SUBSCRIPTIONS_FILE" # Clean up any empty lines

        # Define the ccd file path
        local ccd_file="${OPENVPN_CCD_DIR}/${cert_name}"

        # Remove the client ccd file
        [[ -f "$ccd_file" ]] && rm "$ccd_file"

        log_message "Client $cert_name removed successfully."
    fi

    # Update rules
    if [[ "$action" != "list" ]]; then
        update_subscriptions
        update_client_mapping
        setup_bandwidth_limits
    fi
}

# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------

# Displays the menu header
menu_header() {
    local menu="$1"

    clear
    echo "OpenVPN Traffic Control - $menu"
    echo "------------------------------------"
}

# Create the required folders and files
setup_file_structure() {
    log_message "Creating file structure..."

    # Create OPENVPN_CCD_DIR
    if [ ! -d "$OPENVPN_CCD_DIR" ]; then
        log_message "Creating OpenVPN CCD directory with permissions 0755..."
        mkdir -p "$OPENVPN_CCD_DIR" && chmod 0755 "$OPENVPN_CCD_DIR"
    fi

    # Create BASE_DIR
    if [ ! -d "$SCRIPT_DATA_DIR" ]; then
        log_message "Creating base directory..."
        mkdir -p "$SCRIPT_DATA_DIR"
    fi

    # Create SUBSCRIPTIONS_FILE
    if [ ! -f "$SUBSCRIPTIONS_FILE" ]; then
        log_message "Creating subscription file..."
        touch "$SUBSCRIPTIONS_FILE"
    fi

    # Create TRAFFIC_LOG
    if [ ! -f "$TRAFFIC_LOG" ]; then
        log_message "Creating traffic log file..."
        touch "$TRAFFIC_LOG"
    fi

    # Create QUOTA_HISTORY
    if [ ! -f "$QUOTA_HISTORY" ]; then
        log_message "Creating quota history file..."
        touch "$QUOTA_HISTORY"
    fi

    log_message "File structure created."
}

# Setup the OpenVPN server configuration and patches
setup_openvpn() {
    local hostname="$1"
    local cert_name="$2"

    # Run auto setup script
    setup_openvpn_server "$hostname" "$cert_name"

    # Apply additional patches
    patch_openvpn_config
    patch_openvpn_service
}

# Initial setup chain
initial_setup() {
    local hostname="$1"
    local cert_name="$2"

    # Pre-setup for subscription management
    setup_file_structure

    # OpenVPN setup
    setup_openvpn "$hostname" "$cert_name"

    # Add client to subscriptions
    echo "$cert_name=unlimited" >>"$SUBSCRIPTIONS_FILE"

    # Setup subscription management
    setup_subscriptions
    update_subscriptions
    update_client_mapping
    setup_bandwidth_limits
    setup_cron
    restart_services
}

# Log traffic usage, reset quotas, and update quota limits
update_limits_quotas() {
    update_subscriptions
    update_client_mapping
    setup_bandwidth_limits
    log_traffic_usage
    reset_log_quotas
    handle_limits_quotas "update"
}

# --------------------------------------------------------------------------------------
# Interactive Mode
# --------------------------------------------------------------------------------------

# Interactive initial setup
interactive_setup() {
    while true; do
        menu_header "Setup"

        read -r -p "Enter hostname: " hostname
        read -r -p "Enter certificate name: " cert_name

        initial_setup "$hostname" "$cert_name"

        wait_for_keypress
        break
    done
}

# Interactive SSH restriction
interactive_ssh_restrict() {
    while true; do
        menu_header "SSH"

        echo "1) Restrict access to IP address"
        echo "2) Back to Main Menu"

        echo -e "\nSelect an option: \c"
        read -r ssh_choice

        case "$ssh_choice" in
        1)
            # Loop until a valid IP is provided
            while true; do
                # Prompt for custom IP
                echo -e "\nEnter a custom IP: \c"
                read -r custom_ip

                # Check if the input is not empty
                if [ -z "$custom_ip" ]; then
                    echo "Error: IP address cannot be empty. Please enter a valid IP."
                    wait_for_keypress
                else
                    # Ask for confirmation
                    while true; do
                        echo -e "\nWarning: Restricting SSH access to the specified IP may result in loss of connectivity if the IP is incorrect."
                        echo "Are you sure you want to proceed with restricting access to IP: $custom_ip? (Y/N): \c"
                        read -r ip_choice

                        # Convert choice to uppercase
                        case "${ip_choice^^}" in
                        Y)
                            restrict_ssh_access "$custom_ip"

                            echo "SSH successfully restricted to $custom_ip."
                            wait_for_keypress
                            break 2
                            ;;
                        N)
                            break
                            ;;
                        *)
                            echo "Invalid response. Please enter 'Y' or 'N'."
                            wait_for_keypress
                            ;;
                        esac
                    done
                fi

            done
            ;;
        2)
            break # Return to Main Menu
            ;;
        *)
            echo "Invalid option. Please try again."
            wait_for_keypress
            ;;
        esac
    done
}

# Interactive traffic quota management
interactive_quota_management() {
    while true; do
        menu_header "Traffic Quota Management"

        echo "1) View Traffic Quotas"
        echo "2) Update Traffic Quotas"
        echo "3) Back to Main Menu"

        echo -e "\nSelect an option: \c"
        read -r quota_choice

        case "$quota_choice" in
        1)
            handle_limits_quotas "view"
            wait_for_keypress
            break
            ;;
        2)
            update_limits_quotas
            wait_for_keypress
            break
            ;;
        3)
            break # Return to Main Menu
            ;;
        *)
            echo "Invalid option. Please try again."
            wait_for_keypress
            ;;
        esac
    done
}

# Interactive client management
interactive_client_management() {
    while true; do
        menu_header "Client Management"

        echo "1) List Clients"
        echo "2) Add Client"
        echo "3) Update Client"
        echo "4) Remove Client"
        echo "5) Back to Main Menu"

        echo -e "\nSelect an option: \c"
        read -r client_choice

        case "$client_choice" in
        1)
            manage_client "list"
            wait_for_keypress
            break
            ;;
        2)
            read -r -p "Enter client certificate name: " cert_name

            # Present subscription options
            echo "Select a subscription type:"

            select subscription in "${SUBSCRIPTION_LIST[@]}"; do
                if [[ -n "$subscription" ]]; then
                    break
                else
                    echo "Invalid selection. Please select a valid subscription."
                fi
            done

            while true; do
                echo -e "\nYou are about to ADD a new client with the following details:"
                echo "Client Name: $cert_name"
                echo "Subscription: $subscription"
                echo "Proceed? (Y/N): \c"
                read -r add_choice

                case "${add_choice^^}" in
                Y)
                    manage_client "add" "$cert_name" "$subscription"
                    echo "Client successfully added."
                    wait_for_keypress
                    break
                    ;;
                N)
                    break
                    ;;
                *)
                    echo "Invalid response. Please enter 'Y' or 'N'."
                    ;;
                esac
            done
            ;;
        3)
            read -r -p "Enter client certificate name: " cert_name

            # Present subscription options
            echo "Select a subscription type:"

            select subscription in "${SUBSCRIPTION_LIST[@]}"; do
                if [[ -n "$subscription" ]]; then
                    break
                else
                    echo "Invalid selection. Please select a valid subscription."
                fi
            done

            while true; do
                echo -e "\nYou are about to UPDATE the client with the following details:"
                echo "Client Name: $cert_name"
                echo "Subscription: $subscription"
                echo "Proceed? (Y/N): \c"
                read -r update_choice

                case "${update_choice^^}" in
                Y)
                    manage_client "update" "$cert_name" "$subscription"
                    echo "Client successfully updated."
                    wait_for_keypress
                    break
                    ;;
                N)
                    break
                    ;;
                *)
                    echo "Invalid response. Please enter 'Y' or 'N'."
                    ;;
                esac
            done
            ;;
        4)
            read -r -p "Enter client certificate name to remove: " cert_name

            while true; do
                echo -e "\nYou are about to REMOVE the client with the following details:"
                echo "Client Name: $cert_name"
                echo "Proceed? (Y/N): \c"
                read -r remove_choice

                case "${remove_choice^^}" in
                Y)
                    manage_client "remove" "$cert_name"
                    echo "Client successfully removed."
                    wait_for_keypress
                    break
                    ;;
                N)
                    echo "Operation canceled. Returning to menu."
                    wait_for_keypress
                    break
                    ;;
                *)
                    echo "Invalid response. Please enter 'Y' or 'N'."
                    ;;
                esac
            done
            ;;
        5)
            break # Return to Main Menu
            ;;
        *)
            echo "Invalid option. Please try again."
            wait_for_keypress
            ;;
        esac
    done
}

# Interactive menu
interactive_menu() {
    while true; do
        menu_header ""

        echo "1) Initial Setup"
        echo "2) Restrict SSH access"
        echo "3) View and Update Traffic Usage"
        echo "4) Manage Clients"
        echo "5) Exit"

        echo -e "\nSelect an option: \c"
        read -r main_choice

        case "$main_choice" in
        1)
            # Initial Setup Menu
            interactive_setup
            ;;
        2)
            # Restrict access to SSH
            interactive_ssh_restrict
            ;;
        3)
            # Traffic Quota Management
            interactive_quota_management
            ;;
        4)
            # Client Management Menu
            interactive_client_management
            ;;
        5)
            # Exit
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            wait_for_keypress
            ;;
        esac
    done
}

# --------------------------------------------------------------------------------------
# Command-Line Interface Mode
# --------------------------------------------------------------------------------------

# Function to display usage information
show_example_usage() {
    local mode="$1"

    echo "Example usage:"

    if [[ $mode == "all" || $mode == "setup" ]]; then
        echo "   --setup <hostname> <cert_name>               : Initial setup"
    fi

    if [[ $mode == "all" || $mode == "ssh" ]]; then
        echo "   --ssh <client_ip>                            : Restrict SSH access to IP"
    fi

    if [[ $mode == "all" || $mode == "quota" ]]; then
        echo "   --quota view                                 : View current traffic quotas"
        echo "   --quota update                               : Update traffic quotas"
    fi

    if [[ $mode == "all" || $mode == "client" ]]; then
        echo "   --client list                                : List clients"
        echo "   --client add <cert_name> <subscription>      : Add client"
        echo "   --client update <cert_name> <subscription>   : Update subscription"
        echo "   --client remove <cert_name>                  : Remove client"
    fi
}

# Process command line arguments
process_cmd() {
    case "$1" in
    # Initial Setup
    --setup)
        if [[ -z "$2" || -z "$3" ]]; then
            echo -e "Error: Missing options for $1\n"
            show_example_usage "setup"
            exit 1
        fi

        initial_setup "$2" "$3"
        echo "Setup completed successfully."
        ;;

    # Restrict SSH Access
    --ssh)
        if [[ -z "$2" ]]; then
            echo -e "Error: Missing options for $1\n"
            show_example_usage "ssh"
            exit 1
        fi

        restrict_ssh_access "$2"

        echo "SSH access successfully restricted to $2."
        ;;

    # Traffic Quota Management
    --quota)
        if [[ -z "$2" ]]; then
            echo -e "Error: Missing option for $1\n"
            show_example_usage "quota"
            exit 1
        fi

        if [[ "$2" == "view" ]]; then
            handle_limits_quotas "$2"
        fi

        if [[ "$2" == "update" ]]; then
            update_limits_quotas
        fi
        ;;

    # Client Management
    --client)
        if [[ -z "$2" ||
            ("$2" != "list" && "$2" != "add" && "$2" != "update" && "$2" != "remove") ||
            ("$2" == "add" || "$2" == "update") && (-z "$3" || -z "$4") ||
            "$2" == "remove" && -z "$3" ]]; then
            echo -e "Error: Missing or invalid option(s) for $1\n"
            show_example_usage "client"
            exit 1
        fi

        # Check if the subscription parameter is valid
        if [[ "$2" == "add" || "$2" == "update" ]]; then
            subscription="$4"

            # Check if $subscription exists in the SUBSCRIPTION_LIST array
            found=false
            for sub in "${SUBSCRIPTION_LIST[@]}"; do
                if [[ "$sub" == "$subscription" ]]; then
                    found=true
                    break
                fi
            done

            # If subscription is not found in the list, print the error message
            if ! $found; then
                # Build a dynamic error message listing valid options
                valid_options=$(
                    IFS=", "
                    echo "${SUBSCRIPTION_LIST[*]}"
                )

                # Display the error message
                echo "Error: Invalid subscription level '$subscription'."
                echo "Valid options are: $valid_options."
                exit 1
            fi
        fi

        manage_client "$2" "$3" "$4"
        ;;

    # Default: Invalid action passed
    *)
        show_example_usage "all"
        exit 1
        ;;
    esac
}

# --------------------------------------------------------------------------------------
# Init Script
# --------------------------------------------------------------------------------------

if [[ -z "$1" ]]; then
    # Interactive
    interactive_menu
else
    # CLI
    process_cmd "$@"
fi
