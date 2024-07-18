#date: 2024-07-18T16:49:40Z
#url: https://api.github.com/gists/83182384efeae79c90707730a092f1dc
#owner: https://api.github.com/users/johnsenner

#!/bin/bash

# Set the directory where the screenshots will be saved
SAVE_DIR=~/Desktop/screenshots

# Create the directory if it doesn't exist
mkdir -p $SAVE_DIR

# Loop to take screenshots every 3 seconds
while true; do
    # Get the current timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # Take a screenshot and save it to the device's storage
    adb shell screencap -p /sdcard/screenshot_$TIMESTAMP.png
    
    # Pull the screenshot from the device to the local machine
    adb pull /sdcard/screenshot_$TIMESTAMP.png $SAVE_DIR/screenshot_$TIMESTAMP.png
    
    # Remove the screenshot from the device's storage
    adb shell rm /sdcard/screenshot_$TIMESTAMP.png
    
    # Wait for 3 seconds
    sleep 3
done
