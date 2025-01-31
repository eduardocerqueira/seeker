#date: 2025-01-31T16:51:56Z
#url: https://api.github.com/gists/db304f7a27589defe2912eac37bc33b4
#owner: https://api.github.com/users/numpde

#!/bin/bash

set -euo pipefail

# ASUS ROG FALCHION USB Vendor and Product ID
VENDOR_ID="0b05"
PRODUCT_ID="193e"

# Udev rule file path
RULE_FILE="/etc/udev/rules.d/99-rog-falchion.rules"

echo "Disabling USB autosuspend for ASUS ROG FALCHION (ID $VENDOR_ID:$PRODUCT_ID)..."

# Check if the rule already exists
if grep -q "$VENDOR_ID.*$PRODUCT_ID" "$RULE_FILE" 2>/dev/null; then
    echo "Udev rule already exists. Skipping update."
else
    echo 'ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="'$VENDOR_ID'", ATTRS{idProduct}=="'$PRODUCT_ID'", ATTR{power/control}="on"' | sudo tee "$RULE_FILE" > /dev/null
    echo "Udev rule added: $RULE_FILE"
fi

# Reload udev rules
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Find the correct device path dynamically
DEVICE_PATH=$(ls -d /sys/bus/usb/devices/* | while read -r dir; do
    if [[ -f "$dir/idProduct" ]] && [[ -f "$dir/idVendor" ]]; then
        if [[ "$(cat "$dir/idProduct")" == "$PRODUCT_ID" ]] && [[ "$(cat "$dir/idVendor")" == "$VENDOR_ID" ]]; then
            echo "$dir"
            break
        fi
    fi
done)

if [[ -n "$DEVICE_PATH" ]]; then
    AUTOSUSPEND_STATE=$(cat "$DEVICE_PATH/power/control")
    echo "Current autosuspend state: $AUTOSUSPEND_STATE"
    if [[ "$AUTOSUSPEND_STATE" == "on" ]]; then
        echo "✔ USB autosuspend is disabled for ASUS ROG FALCHION."
    else
        echo "⚠ Warning: USB autosuspend may not be disabled. Check manually."
    fi
else
    echo "⚠ Device not found. The rule will apply when the keyboard is reconnected."
fi
