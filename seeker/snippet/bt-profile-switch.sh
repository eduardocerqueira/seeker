#date: 2023-07-03T17:04:48Z
#url: https://api.github.com/gists/8cb2616d7ca11aa5d5bb177fbeb7421e
#owner: https://api.github.com/users/Lucs1590

#!/bin/bash

CARD=$(pactl list | grep bluez_card | awk '{print $NF}')
BLUETOOTH_DEVICE=$(pacmd list-sinks | grep -o '<bluez_sink[^>]*' | awk -F "<" '{print $2}' | awk -F ">" '{print $1}')

PROFILE_A2DP="a2dp_sink"
PROFILE_HEADSET_UNIT="handsfree_head_unit"

set_profile() {
    local profile=$1
    local card=$2
    local sink=$3
    if ! pactl set-card-profile "$card" "$profile"; then
        echo "Error: Failed to set card profile to '$profile'"
        exit 3
    fi
    if ! pacmd set-default-sink "$sink"; then
        echo "Error: Failed to set default sink to '$sink'"
        exit 4
    fi
}

if [ "${1:-}" = "listen" ]; then
    set_profile "$PROFILE_A2DP" "$CARD" "$BLUETOOTH_DEVICE"
    exit
elif [ "${1:-}" = "speak" ]; then
    set_profile "$PROFILE_HEADSET_UNIT" "$CARD" "$BLUETOOTH_DEVICE"
else
    echo "Error: Unsupported option '$1'"
    exit 1
fi
