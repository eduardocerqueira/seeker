#date: 2024-12-20T16:54:47Z
#url: https://api.github.com/gists/fb27b69618b5f7876041c4178ada441e
#owner: https://api.github.com/users/salah-rashad

#!/bin/bash

# Set the volume of the paplay command to 7%
ALARM_VOLUME=0.07

battery=""
output=""
online=""

update() {
    # Get battery percentage
    battery=$(upower -i /org/freedesktop/UPower/devices/battery_BAT0 | grep -E "percentage" | awk '{print $2}' | sed 's/%//')
    # Get battery state
    output=$(upower -i /org/freedesktop/UPower/devices/line_power_AC)
    # [yes]: charging, [no]: discharging
    online=$(echo "$output" | grep -m 1 -o "online:[[:space:]]*\(yes\|no\)" | awk '{print $2}')
}

while true; do
    # Initialize battery info
    update

    # Check battery level and show notification or alarm
    if (("$battery" >= 95)); then
        # Show alarm and keep notifying until battery is below 95% or charger is disconnected
        while (("$battery" >= 95)) && [ "$online" == "yes" ]; do
            notify-send -i battery-full "Battery Level [$battery%]" "Please unplug the charger."
            pactl set-sink-volume "$(pactl list short sinks | awk '$2 == "alsa_output.pci-0000_00_1f.3.analog-stereo" { print $1 }')" "${ALARM_VOLUME}"
            paplay /home/salah/Music/best_alarm.ogg
            sleep 10
            update
        done
    elif (("$battery" >= 90)); then
        while (("$battery" >= 90)) && [ "$online" == "yes" ]; do
            notify-send -i battery-good "Battery Level [$battery%]" "Please unplug the charger."
            sleep 15
            update
        done
    elif (("$battery" <= 30)); then
        while (("$battery" <= 30)) && [ "$online" == "no" ]; do
            notify-send -i battery-caution "⚠️ Battery Level [$battery%]" "Please plug in the charger."
            pactl set-sink-volume "$(pactl list short sinks | awk '$2 == "alsa_output.pci-0000_00_1f.3.analog-stereo" { print $1 }')" "${ALARM_VOLUME}"
            paplay /home/salah/Music/low_battery.ogg
            sleep 10
            update
        done
    elif (("$battery" <= 35)); then
        while (("$battery" <= 35)) && (("$battery" > 30)) && [ "$online" == "no" ]; do
            notify-send -i battery-low "Battery Level [$battery%]" "Please plug in the charger."
            sleep 15
            update
        done
    fi

    # Sleep for 30 seconds
    sleep 30
done
