#date: 2025-12-26T16:23:05Z
#url: https://api.github.com/gists/1b3131d0bdddd37968cf81270eecef46
#owner: https://api.github.com/users/Harsh-cyber005

#!/bin/bash

set -euo pipefail

VM_ID_FILE="/var/lib/vm-monitor/vm-id"

if [ ! -s "$VM_ID_FILE" ]; then
        echo "ERROR: VM ID file missing or empty at $VM_ID_FILE" >&2
        exit 1
fi

VM_ID=$(cat "$VM_ID_FILE")

CPU_USED=$(top -bn1 | awk -F',' '/Cpu/ {print $4}' | awk '{print 100-$1}')

read RAM_TOTAL RAM_USED <<< $(free -m | awk '/Mem:/ {print $2, $3}')

read DISK_TOTAL DISK_USED <<< $(df -m --output=size,used / | tail -1)

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

HOSTNAME=$(hostname -I | awk '{print $1}')

IP=$(curl -s -4 https://icanhazip.com | tr -d '\n')

JSON=$(printf '{\n'
printf '\t"vmId":"%s",\n' "$VM_ID"
printf '\t"status":"%s",\n' "running"
printf '\t"timestamp":"%s",\n' "$TS"
printf '\t"ramUsedMB":%d,\n' "$RAM_USED"
printf '\t"ramTotalMB":%d,\n' "$RAM_TOTAL"
printf '\t"diskUsedMB":%d,\n' "$DISK_USED"
printf '\t"diskTotalMB":%d,\n' "$DISK_TOTAL"
printf '\t"cpuUsed":%.2f,\n' "$CPU_USED"
printf '\t"hostname":"%s",\n' "$HOSTNAME"
printf '\t"publicIp":"%s"\n' "$IP"
printf '}\n')

echo ${JSON}

curl -sS --fail --connect-timeout 3 --max-time 5 -X POST -H "Content-Type: application/json" --data "$JSON" http://localhost:5000/monitor/vm-status > /dev/null