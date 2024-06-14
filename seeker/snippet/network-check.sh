#date: 2024-06-14T17:10:00Z
#url: https://api.github.com/gists/899c2f57991a25d012cb7dcd273ec353
#owner: https://api.github.com/users/qailanet

#!/bin/bash

# Define the network interface name
interface="enx00e04c680059"

# Log directory
log_dir="/root/scripts/network-check/network_logs"

# Ensure the log directory exists
mkdir -p "$log_dir"

# Log file with today's date
LOG_FILE="$log_dir/network_check_$(date +"%Y-%m-%d").log"

# Function to perform log rotation
rotate_logs() {
  find "$log_dir" -name "network_check_*.log" -mtime +7 -exec rm {} \;
}


# Check if the network interface is up
if ip link show dev "$interface" | grep -q "UP,LOWER_UP"; then
  echo "Network interface $interface is already up."
  #echo "$(date) - Network interface $interface is already up." >> "$LOG_FILE"
else
  # Bring the network interface up
  ip link set dev "$interface" up
  if [ $? -eq 0 ]; then
    echo "Network interface $interface has been brought up successfully."
    echo "$(date) - Network interface $interface has been brought up successfully." >> "$LOG_FILE"
    sleep 15
    # Run vtysh command and log the output
    vtysh_output=$(vtysh -c 'show ipv6 ospf6 neighbor')
    echo "vtysh output:" >> "$LOG_FILE"
    echo "$vtysh_output" >> "$LOG_FILE"
  else
    echo "Failed to bring up network interface $interface."
    echo "$(date) - Failed to bring up network interface $interface." >> "$LOG_FILE"
  fi
fi


# Perform log rotation
rotate_logs