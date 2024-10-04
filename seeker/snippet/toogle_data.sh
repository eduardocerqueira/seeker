#date: 2024-10-04T16:44:17Z
#url: https://api.github.com/gists/4150e2e03e1f60c20c657e7e0b3f75d2
#owner: https://api.github.com/users/Imgkl

#!/bin/bash

# Function to check if the device can reach the internet by pinging Google
check_internet() {
    adb shell ping -c 1 8.8.8.8 > /dev/null 2>&1
    return $?
}

# Get the current internet connection status by pinging Google
if check_internet; then
    echo "Device is connected to the internet. Disabling the data connection..."
    adb shell svc data disable
    echo "Data connection has been disabled."
else
    echo "Data connection has been disabled."
    adb shell svc data enable
    echo "Device is connected to the internet."
fi
