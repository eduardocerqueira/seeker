#date: 2024-04-08T16:55:29Z
#url: https://api.github.com/gists/07457e4b76d9668a189ba24dfbcf24f7
#owner: https://api.github.com/users/fmg-cconley

#!/bin/bash

# Used if you want to log your memory values over time to see the behaviour.

csv_file="output.csv"

# Create the CSV file if it doesn't exist
if [[ ! -f "$csv_file" ]]; then
    echo "Seconds Since Starting,Used Memory (MB),Cached Memory (MB),Free Memory (MB),1 Min CPU,5 Min CPU,15 Min CPU,Total Memory Available (MB),Total Memory Used by vmmemwsl (MB)" > "$csv_file"
fi

# Define conversion constants
MB=$((1024 * 1024))

# Start time
start_time=$(date +%s)

while true; do
    echo "Collecting system information..."

    # Get system memory information in bytes
    mem_info=$(free -b | grep "Mem:")
    used_mem=$(echo "$mem_info" | awk '{print $3}')
    cached_mem=$(echo "$mem_info" | awk '{print $6}')
    free_mem=$(echo "$mem_info" | awk '{print $4}')

    # Convert memory values to megabytes
    used_mem_mb=$((used_mem / MB))
    cached_mem_mb=$((cached_mem / MB))
    free_mem_mb=$((free_mem / MB))

    # Get CPU load averages
    load_1min=$(cat /proc/loadavg | awk '{print $1}')
    load_5min=$(cat /proc/loadavg | awk '{print $2}')
    load_15min=$(cat /proc/loadavg | awk '{print $3}')

    # Get total memory available in Windows
    echo "Fetching total memory available in Windows..."
    total_mem_avail=$(powershell.exe "(Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory" | sed 's/\r$//')

    # Convert total memory available to megabytes
    total_mem_avail_mb=$((total_mem_avail / 1024))

    # Get total memory used by vmmemwsl process
    echo "Fetching total memory used by vmmemwsl process..."
    total_mem_vmmem=$(powershell.exe "(Get-Process -Name vmmemwsl | Measure-Object -Property WS -Sum).Sum" | sed 's/\r$//')

    # Convert total memory used by vmmemwsl to megabytes
    total_mem_vmmem_mb=$((total_mem_vmmem / MB))

    # Calculate seconds since starting
    current_time=$(date +%s)
    seconds_since_start=$((current_time - start_time))

    # Append the values to the CSV file
    echo "Appending system information to $csv_file..."
    echo "$seconds_since_start,$used_mem_mb,$cached_mem_mb,$free_mem_mb,$load_1min,$load_5min,$load_15min,$total_mem_avail_mb,$total_mem_vmmem_mb" >> "$csv_file"

    echo "System information collected and logged successfully."

    # Wait for 1 second before the next iteration
    echo "Waiting for the next iteration..."
    sleep 10
done