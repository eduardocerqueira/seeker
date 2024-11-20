#date: 2024-11-20T17:02:45Z
#url: https://api.github.com/gists/e7be0bf74fed14b51a10c426c55b4884
#owner: https://api.github.com/users/natemollica-nm

#!/bin/sh

set -e

# Function to display usage
usage() {
  echo "Usage: $0"
  echo "This script measures connection times to all Consul servers in milliseconds."
  exit 1
}

# Function to check latency metrics for a given server IP
check_latency() {
  server_ip="$1"
  echo "Testing latency to $server_ip"

  # Collect curl timing metrics in seconds
  metrics=$(curl -o /dev/null -sk -w '%{time_connect} %{time_starttransfer} %{time_total}' "https://$server_ip:8501/v1/status/leader")

  # Extract metrics and convert to milliseconds
  time_connect=$(echo "$metrics" | awk '{printf "%.0f", $1 * 1000}')
  time_starttransfer=$(echo "$metrics" | awk '{printf "%.0f", $2 * 1000}')
  time_total=$(echo "$metrics" | awk '{printf "%.0f", $3 * 1000}')

  # Display the metrics in milliseconds
  echo "Server: $server_ip"
  echo "  Connect Time: $time_connect ms"
  echo "  Start Transfer Time: $time_starttransfer ms"
  echo "  Total Time: $time_total ms"
  echo ""
}

# Ensure Consul CLI is available
if ! command -v consul >/dev/null 2>&1; then
  echo "Error: Consul CLI not found. Please install and configure it before running this script."
  exit 1
fi

# Get the list of Consul servers and their IPs
echo "Fetching list of Consul server IPs..."
server_ips=$(consul members | awk '/server/ {print $2}' | cut -d':' -f1)

if [ -z "$server_ips" ]; then
  echo "No servers found in Consul cluster."
  exit 1
fi

# Iterate over each server IP and check latency
echo "Measuring latency for the following servers (in milliseconds):"
echo "$server_ips"
echo ""

for server_ip in $server_ips; do
  check_latency "$server_ip"
done

echo "Latency checks completed."
