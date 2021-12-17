#date: 2021-12-17T17:08:57Z
#url: https://api.github.com/gists/a614eda4749ffcb3cc099d8477296e8a
#owner: https://api.github.com/users/gpazuch

# Corsair headsets will stuck the apps on your linux system. This is due to wrong usb-mapping.

# thx to http://www.c0urier.net/2016/corsair-gaming-void-usb-rgb-linux-fun

# 1. open terminal
# 2. type this and search the line with your headset
lsusb


# Get the USB ID of the headset and add it to xorg.conf:

sudo nano /etc/X11/xorg.conf

# in my case it looks like this:
Section "InputClass"
    Identifier "Corsair"
    MatchUSBID "1b1c:1b2a"        <----    replace the id with yours
    Option "StartKeysEnabled" "False"
    Option "StartMouseEnabled" "False"
EndSection

# restart system
shutdown -r now
