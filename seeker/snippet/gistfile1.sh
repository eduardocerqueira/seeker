#date: 2023-02-20T17:01:59Z
#url: https://api.github.com/gists/8ec737e0331d73f0c6fb972fe768fd52
#owner: https://api.github.com/users/inside36com

# use lsusb to find the details of the serial adapter to create a udev rule

~$ lsusb | grep Serial
Bus 003 Device 002: ID 067b:2303 Prolific Technology, Inc. PL2303 Serial Port


# based on the output from lsusb add a rule to /etc/udev/rules.d/local.rules

~$ cat /etc/udev/rules.d/local.rules
#Bus 004 Device 002: ID 067b:2303 Prolific Technology, Inc. PL2303 Serial Port"
ENV{ID_BUS}=="usb", ENV{ID_VENDOR_ID}=="067b", ENV{ID_MODEL_ID}=="2303", SYMLINK+="apc-usb"

##########

# set /etc/apcupsd.conf to use the new device
~$ head /etc/apcupsd.conf
UPSCABLE smart
UPSTYPE apcsmart
DEVICE /dev/apc-usb

##########

# After a reboot check the status of the UPS using
~$ sudo apcaccess status