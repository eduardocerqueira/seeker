#date: 2022-04-06T17:06:38Z
#url: https://api.github.com/gists/b68b0d7ecf8ec35a4d1ee7a9ea8552b8
#owner: https://api.github.com/users/gwisp2

#!/usr/bin/env sh

set -e

print_warning() {
    echo -e "\033[1;33mentrypoint.sh: $*\033[0m"
}

print_error() {
    echo -e "\033[0;31mentrypoint.sh: $*\033[0m"
}

abort()
{
    print_error "$*" >&2
    exit 1
}

if [ ! -z "$RTL_DEVICE" ]; then
    if [ ! -e "$RTL_DEVICE" ]; then
      abort "$RTL_DEVICE does not exist"
    fi

    # $RTL_DEVICE may be a symlink created by udev. However rtl433 is not smart enough to find such device
    # if corresponding /dev/bus/usb/XXX/YYY is not exposed to the container. 
    # Workaround: create a symlink from /dev/bus/usb/XXX/YYY to $RTL_DEVICE.
    # See https://github.com/hertzg/rtl_433_docker/issues/14 for the discussion.
    REAL_DEVICE_NAME="/dev/$(udevadm info "--name=$RTL_DEVICE" -q name)" || abort "failed to find real device name for $RTL_DEVICE"
    if [ "$RTL_DEVICE" != "$REAL_DEVICE_NAME" ] && [ ! -f "$REAL_DEVICE_NAME" ]; then
        mkdir -p "$(dirname "$REAL_DEVICE_NAME")"
        ln -s "$RTL_DEVICE" "$REAL_DEVICE_NAME"
    fi
elif [ ! -d /dev/bus/usb ]; then
    print_warning "RTL_DEVICE is not set and /dev/bus/usb does not exist. Did you forget to expose device into a container?"
fi

/usr/local/bin/rtl_433 "$@"