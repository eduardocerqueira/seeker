#date: 2022-11-07T17:08:39Z
#url: https://api.github.com/gists/82c3af9fa59994b91528e4972edd0162
#owner: https://api.github.com/users/Howard20181

#!/bin/sh
changeWLAN() {
    adb wait-for-device
    case "$1" in
    enable)
        if [ "$(adb shell settings get global wifi_on)" = "0" ]; then
            if (adb shell svc wifi enable); then
                logger -p info -t adb-changeWLAN 'WLAN enabled'
            else
                logger -p err -t adb-changeWLAN 'failed to enable WLAN'
            fi
        else
            logger -p info -t adb-changeWLAN 'WLAN already enabled'
        fi
        ;;
    disable)
        if [ "$(adb shell settings get global wifi_on)" = "1" ]; then
            if (adb shell svc wifi disable); then
                logger -p info -t adb-changeWLAN 'WLAN disabled'
            else
                logger -p err -t adb-changeWLAN 'failed to disable WLAN'
            fi
        else
            logger -p info -t adb-changeWLAN 'WLAN already disabled'
        fi
        ;;
    esac
}
changeUSBnet() {
    adb wait-for-device
    case "$1" in
    enable)
        if [ "$(adb shell svc usb getFunctions)" != "rndis" ]; then
            if (changeWLAN disable); then
                if (adb shell svc usb setFunctions rndis); then
                    logger -p info -t adb-changeUSBnet 'USB tethering enabled'
                else
                    logger -p err -t adb-changeUSBnet 'failed to enable USB tethering'
                fi
            fi
        else
            logger -p info -t adb-changeUSBnet 'USB tethering already enabled'
        fi
        ;;
    disable)
        if [ "$(adb shell svc usb getFunctions)" = "rndis" ]; then
            if (adb shell svc usb setFunctions); then
                logger -p info -t adb-changeUSBnet 'USB tethering disabled'
                (changeWLAN enable) &
            else
                logger -p err -t adb-changeUSBnet 'failed to disable USB tethering'
            fi
        else
            logger -p info -t adb-changeUSBnet 'USB tethering already disabled'
        fi
        ;;
    esac
}

if [ "$ACTION" = "connected" ] && [ "$INTERFACE" = "wan" ]; then
    (changeUSBnet disable) &
fi
if [ "$ACTION" = "disconnected" ] || [ "$ACTION" = "disconnecting" ] && [ "$INTERFACE" = "wan" ]; then
    (changeUSBnet enable) &
fi
