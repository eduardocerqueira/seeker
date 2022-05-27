#date: 2022-05-27T17:09:35Z
#url: https://api.github.com/gists/a4255461587c07f4bb5a3f0610cdf1e7
#owner: https://api.github.com/users/NameOfTheDragon

cd ~/klipper
make
~/klipper/scripts/flash-sdcard.sh /dev/serial/by-id/usb-Klipper_lpc1769_0DF0FF0EA69869AF2046415EC42000F5-if00 btt-skr-turbo-v1.4
~/klipper/scripts/flash-sdcard.sh /dev/serial/by-id/usb-Klipper_lpc1769_10D0FF0BA39869AFC04E405EC12000F5-if00 btt-skr-turbo-v1.4
cd ~
