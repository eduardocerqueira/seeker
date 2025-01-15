#date: 2025-01-15T17:11:55Z
#url: https://api.github.com/gists/3d0f9c6d6a5f1ab6b84938e3d06f3d55
#owner: https://api.github.com/users/celeroncoder

#!/bin/bash

# Thresholds
LOW_BATTERY_THRESHOLD=30
UNPLUG_THRESHOLD=80
NOTIFICATION_INTERVAL=10

# Track the last notified percentage
last_notified_percentage=100

while true; do
    # Get battery status and percentage
    battery_info=$(acpi -b)
    battery_status=$(echo "$battery_info" | grep -Po '(?<=: )\w+')
    battery_percent=$(echo "$battery_info" | grep -Po '\d+(?=%)')

    # Notify when battery is low
    if [[ $battery_percent -le $LOW_BATTERY_THRESHOLD && ($last_notified_percentage -gt $LOW_BATTERY_THRESHOLD || $last_notified_percentage -eq 100) ]]; then
        notify-send "Battery Low" "Battery is at $battery_percent%!"
        last_notified_percentage=$LOW_BATTERY_THRESHOLD
    fi

    # Notify for every 10% drop after 30%
    if [[ $battery_percent -le $LOW_BATTERY_THRESHOLD && $((last_notified_percentage - battery_percent)) -ge $NOTIFICATION_INTERVAL ]]; then
        notify-send "Battery Low" "Battery is at $battery_percent%!"
        last_notified_percentage=$battery_percent
    fi

    # Notify to unplug when charging and battery reaches 80%
    if [[ $battery_status == "Charging" && $battery_percent -ge $UNPLUG_THRESHOLD && $last_notified_percentage -lt $UNPLUG_THRESHOLD ]]; then
        notify-send "Battery Charged" "Battery is at $battery_percent%. You can unplug the charger."
        last_notified_percentage=$UNPLUG_THRESHOLD
    fi

    # Wait for 1 minute before checking again
    sleep 60
done