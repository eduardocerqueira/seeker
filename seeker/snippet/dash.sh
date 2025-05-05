#date: 2025-05-05T17:13:38Z
#url: https://api.github.com/gists/2daef330021af2a231efdba42523cf8d
#owner: https://api.github.com/users/jvcss

#!/bin/bash

# Ensure the required tools are installed
if ! command -v vmstat &> /dev/null || ! command -v df &> /dev/null; then
  echo "This script requires 'vmstat' and 'df'. Please install them: sudo apt install procps coreutils"
  exit 1
fi

# Define update interval in seconds
INTERVAL=2

# ANSI escape codes for cursor control
MOVE_CURSOR() { echo -en "\033[$1;${2:-1}H"; }
CLEAR_SCREEN() { echo -en "\033[2J"; }
HIDE_CURSOR() { echo -en "\033[?25l"; }
SHOW_CURSOR() { echo -en "\033[?25h"; }

# Function to print the dashboard header
print_header() {
  MOVE_CURSOR 1
  echo "================= SYSTEM USAGE DASHBOARD ================="
  echo "Press Ctrl+C to exit."
  echo "----------------------------------------------------------"
}

# Function to display CPU and RAM usage
display_cpu_ram() {
  MOVE_CURSOR 5
  echo "CPU and RAM Usage:"
  MOVE_CURSOR 6
  echo "CPU Usage (%)      Free RAM (MB)"
  MOVE_CURSOR 7
  vmstat 1 1 | awk 'NR==3 { printf "%-15s %-15s", 100-$15, $4/1024 }'
}

# Function to display disk usage
display_disk_usage() {
  MOVE_CURSOR 10
  echo "Disk Usage:"
  df -h --output=target,pcent | awk 'NR==1 { printf "%-20s %-10s\n", $1, $2 } NR>1 { printf "%-20s %-10s\n", $1, $2 }' | tail -n +2 | while IFS= read -r line; do
    echo "$line"
  done
}

# Main loop
CLEAR_SCREEN
HIDE_CURSOR
trap "CLEAR_SCREEN; SHOW_CURSOR; exit" SIGINT SIGTERM

print_header

while true; do
  display_cpu_ram
  display_disk_usage
  sleep $INTERVAL
done
