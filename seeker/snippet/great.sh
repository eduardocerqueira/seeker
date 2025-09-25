#date: 2025-09-25T17:05:34Z
#url: https://api.github.com/gists/de30456402b58f5f8882b16762231776
#owner: https://api.github.com/users/RajChowdhury240

#!/bin/bash

# DCProtect Status and Activation Info Script

echo "================================================"
echo "DCProtect Information Script"
echo "================================================"
echo ""

# Step 1: Check if DCProtect is running
echo "Step 1: Checking DCProtect process status..."
dcprotect_processes=$(ps aux | grep -i "DCProtect" | grep -v grep)

if [ -n "$dcprotect_processes" ]; then
    echo "DCProtect running"
else
    echo "DCProtect not running"
fi

echo ""

# Step 2: Extract activation information from log file
echo "Step 2: Reading activation information..."
LOG_FILE="/Library/Application Support/DCProtect/Shared/Logs/InstallationTracking.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found at $LOG_FILE"
    exit 1
fi

echo "Log file found. Extracting information..."
echo ""

# Extract values using grep and sed
DEVICE_ID=$(grep -o '<InstalledDeviceId>[^<]*</InstalledDeviceId>' "$LOG_FILE" | sed 's/<InstalledDeviceId>\(.*\)<\/InstalledDeviceId>/\1/' | tail -1)

MACHINE_NAME=$(grep -o '<MachineName>[^<]*</MachineName>' "$LOG_FILE" | sed 's/<MachineName>\(.*\)<\/MachineName>/\1/' | tail -1)

# Extract email using regex - looking for email patterns
EMAIL=$(grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' "$LOG_FILE" | head -1)

# Display results in table format
echo "================================================"
echo "DCPROTECT ACTIVATION INFORMATION"
echo "================================================"

printf "%-20s | %-50s\n" "Field" "Value"
printf "%-20s-+-%-50s\n" "--------------------" "--------------------------------------------------"

if [ -n "$DEVICE_ID" ]; then
    printf "%-20s | %-50s\n" "InstalledDeviceId" "$DEVICE_ID"
else
    printf "%-20s | %-50s\n" "InstalledDeviceId" "Not found"
fi

if [ -n "$MACHINE_NAME" ]; then
    printf "%-20s | %-50s\n" "MachineName" "$MACHINE_NAME"
else
    printf "%-20s | %-50s\n" "MachineName" "Not found"
fi

if [ -n "$EMAIL" ]; then
    printf "%-20s | %-50s\n" "Email" "$EMAIL"
else
    printf "%-20s | %-50s\n" "Email" "Not found"
fi

echo "================================================"

# Optional: Show raw XML snippets for verification
echo ""
echo "Raw XML data found:"
echo "-------------------"

if [ -n "$DEVICE_ID" ]; then
    echo "Device ID XML: $(grep -o '<InstalledDeviceId>[^<]*</InstalledDeviceId>' "$LOG_FILE" | tail -1)"
fi

if [ -n "$MACHINE_NAME" ]; then
    echo "Machine Name XML: $(grep -o '<MachineName>[^<]*</MachineName>' "$LOG_FILE" | tail -1)"
fi

if [ -n "$EMAIL" ]; then
    echo "Email found in log: $EMAIL"
fi