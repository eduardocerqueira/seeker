#date: 2023-01-09T17:08:25Z
#url: https://api.github.com/gists/ec201c10a6da8f53874e354145fbf02a
#owner: https://api.github.com/users/rodan

#!/bin/bash

# NAME
#       usb_control.sh - enable/disable usb devices via bus bind/unbind commands
#
# SYNOPSIS
#       usb_control.sh [-v VENDOR_ID] [-p PRODUCT_ID] [-e] [-d] [-i] [-h]
#
# OPTIONS
#       -v VENDOR_ID, -p PRODUCT_ID
#           mandatory usb device identifiers
#       -e, -d
#           enable/disable device
#       -i
#           show bus identifier where the device has been found
#
# AUTHOR
#       Petre Rodan, 2022 https://gist.github.com/rodan
#

show_usage() {
    echo 'Usage:'
    echo "    $0 [-v VENDOR_ID] [-p PRODUCT_ID] [-e] [-d] [-i]"
    echo ''
    echo ' Options:'
    echo '   -v VENDOR_ID       specify usb vendor identifier (in 4 digit hex notation)'
    echo '   -p PRODUCT_ID      specify usb product identifier (in 4 digit hex notation)'
    echo '   -i                 get usb bus id for VENDOR_ID:PRODUCT_ID'
    echo '   -e                 enable device defined by VENDOR_ID:PRODUCT_ID'
    echo '   -d                 disable device defined by VENDOR_ID:PRODUCT_ID'
}

#VENDOR='0909'
#PRODUCT='001c'

dev_detect() {
    find /sys/bus/usb/devices/ -maxdepth 1 -type l | while read -r link; do 
        if [[ -f "${link}/idVendor" && -f "${link}/idProduct" && $(cat "${link}/idVendor") == "${VENDOR}" && $(cat "${link}/idProduct") == "${PRODUCT}" ]]; then 
            basename "${link}"
        fi
    done
}

dev_disable() {
    DEVICE="$(dev_detect)"
    [ -n "${DEVICE}" ] && {
        echo "device found under usb bus id ${DEVICE}"
        echo "${DEVICE}" > /sys/bus/usb/drivers/usb/unbind
    }
}

dev_enable() {
    DEVICE="$(dev_detect)"
    [ -n "${DEVICE}" ] && {
        echo "device found under usb bus id ${DEVICE}"
        echo "${DEVICE}" > /sys/bus/usb/drivers/usb/bind
    }
}

opt_disable=false
opt_enable=false
opt_info=false

while (( "$#" )); do
    if [ "$1" = "-v" ]; then
        VENDOR=$2
        shift;
        shift;
    elif [ "$1" = "-p" ]; then
        PRODUCT=$2
        shift;
        shift;
    elif [ "$1" = "-d" ]; then
        opt_disable=true
        shift;
    elif [ "$1" = "-e" ]; then
        opt_enable=true
        shift;
    elif [ "$1" = "-i" ]; then
        opt_info=true
        shift;
    else
        show_usage
    fi
done

if [[ -z "${VENDOR}" || -z "${PRODUCT}" ]]; then
    show_usage
    exit 1
fi

if ${opt_info}; then
    dev_detect
elif ${opt_disable}; then
    dev_disable
elif ${opt_enable}; then
    dev_enable
else
    show_usage
fi
