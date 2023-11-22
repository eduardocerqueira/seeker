#date: 2023-11-22T16:44:57Z
#url: https://api.github.com/gists/1e51e86def73c83dff4f493e7baff5f2
#owner: https://api.github.com/users/albeec13

# Add this line to crontab for the pi user so it runs at bootup with retropie (remove leading # of course,
# and change path to wherever you decide to put this script)
# @reboot /usr/bin/bash /home/pi/bluetoothReconnect.sh >/dev/null 2>&1 &

# Replace with your controllers' Bluetooth IDs
BT1="EC:B0:65:BF:65:05"
BT2="ED:C9:39:85:26:64"

while true; do
    BT1_AVAIL=$(bluetoothctl info ${BT1} | awk '{print $3 $4}')
    BT2_AVAIL=$(bluetoothctl info ${BT2} | awk '{print $3 $4}')

    if [ "${BT1_AVAIL}" == "notavailable" ] || [ "${BT2_AVAIL}" == "notavailable" ]; then
        echo "Starting scan (timeout in 10s):"
        bluetoothctl --timeout 10 scan on &
    fi

    BT1_FOUND=$(bluetoothctl info ${BT1} | grep Connected | awk '{print $2}')
    BT2_FOUND=$(bluetoothctl info ${BT2} | grep Connected | awk '{print $2}')

    if [ "${BT1_FOUND}" == "no" ]; then
        bluetoothctl connect ${BT1}
    fi

    if [ "${BT2_FOUND}" == "no" ]; then
        bluetoothctl connect ${BT2}
    fi

    sleep 12
done