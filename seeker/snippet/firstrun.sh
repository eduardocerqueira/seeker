#date: 2025-11-17T16:49:59Z
#url: https://api.github.com/gists/5f2f7980301daab1caaae63993816179
#owner: https://api.github.com/users/lbt-boa

#!/bin/bash
#
# This file is based on the v2 rpi-imager
#
# Main concern is that it seems to be using /boot/ and not /boot/firmware/
# However it works - probably due to firstboot stuff
#
# Use /usr/lib/raspberrypi-sys-mods and only fallback to for older
# versions of the OS


# we sed __NEWHOST__ to the hostname in Raspberry.org :)

set +e

CURRENT_HOSTNAME=$(cat /etc/hostname | tr -d " \t\
\r")
if [ -f /usr/lib/raspberrypi-sys-mods/imager_custom ]; then
   /usr/lib/raspberrypi-sys-mods/imager_custom set_hostname __NEWHOST__
else
   echo __NEWHOST__ >/etc/hostname
   sed -i "s/127.0.1.1.*$CURRENT_HOSTNAME/127.0.1.1\t__NEWHOST__/g" /etc/hosts
fi
FIRSTUSER=$(getent passwd 1000 | cut -d: -f1)
FIRSTUSERHOME=$(getent passwd 1000 | cut -d: -f6)
if [ -f /usr/lib/raspberrypi-sys-mods/imager_custom ]; then
   /usr/lib/raspberrypi-sys-mods/imager_custom enable_ssh -k 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAfdRRfvmapi0MeD7FWNhu/Q2DGjB0lbRcCdF4tr+vIN david@ash'
else
   install -o "$FIRSTUSER" -m 700 -d "$FIRSTUSERHOME/.ssh"
cat > "$FIRSTUSERHOME/.ssh/authorized_keys" <<'EOF'
########################### Put a pub key here
EOF
   chown "$FIRSTUSER:$FIRSTUSER" "$FIRSTUSERHOME/.ssh/authorized_keys"
   chmod 600 "$FIRSTUSERHOME/.ssh/authorized_keys"
   echo 'PasswordAuthentication no' >>/etc/ssh/sshd_config
   systemctl enable ssh
fi
if [ -f /usr/lib/userconf-pi/userconf ]; then
   /usr/lib/userconf-pi/userconf 'pi' '$y$jB5$BBXJpcbHCFKEss2PmILEJ/$0pPznapuHBhktjV8WYB5ZFX5gI342FEnO1iKzR6HbVB'
else
   echo "$FIRSTUSER:$y$jB5$BBXJpcbHCFKEss2PmILEJ/$0pPznapuHBhktjV8WYB5ZFX5gI342FEnO1iKzR6HbVB" | chpasswd -e
   if [ "$FIRSTUSER" != "pi" ]; then
      usermod -l "pi" "$FIRSTUSER"
      usermod -m -d "/home/pi" "pi"
      groupmod -n "pi" "$FIRSTUSER"
      if grep -q "^autologin-user=" /etc/lightdm/lightdm.conf ; then
         sed /etc/lightdm/lightdm.conf -i -e "s/^autologin-user=.*/autologin-user=pi/"
      fi
      if [ -f /etc/systemd/system/getty@tty1.service.d/autologin.conf ]; then
         sed /etc/systemd/system/getty@tty1.service.d/autologin.conf -i -e "s/$FIRSTUSER/pi/"
      fi
      if [ -f /etc/sudoers.d/010_pi-nopasswd ]; then
         sed -i "s/^$FIRSTUSER /pi /" /etc/sudoers.d/010_pi-nopasswd
      fi
   fi
fi
if [ -f /usr/lib/raspberrypi-sys-mods/imager_custom ]; then
   /usr/lib/raspberrypi-sys-mods/imager_custom set_wlan '<your SSID' '<your key>' 'GB'
else
cat >/etc/wpa_supplicant/wpa_supplicant.conf <<'WPAEOF'
country=GB
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
ap_scan=1

update_config=1
network={
	ssid="SSID"
	psk=YOUR PSK
}
WPAEOF
   chmod 600 /etc/wpa_supplicant/wpa_supplicant.conf
   rfkill unblock wifi
   for filename in /var/lib/systemd/rfkill/*:wlan ; do
       echo 0 > $filename
   done
fi
if [ -f /usr/lib/raspberrypi-sys-mods/imager_custom ]; then
   /usr/lib/raspberrypi-sys-mods/imager_custom set_keymap 'gb'
   /usr/lib/raspberrypi-sys-mods/imager_custom set_timezone 'Europe/London'
else
   rm -f /etc/localtime
   echo "Europe/London" >/etc/timezone
   dpkg-reconfigure -f noninteractive tzdata
cat >/etc/default/keyboard <<'KBEOF'
XKBMODEL="pc105"
XKBLAYOUT="gb"
XKBVARIANT=""
XKBOPTIONS=""

KBEOF
   dpkg-reconfigure -f noninteractive keyboard-configuration
fi

# Extra by lbt
cat > /root/.ssh/authorized_keys <<'EOF'
YOUR PUB KEY
EOF
chown root:root /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

mkdir -p /everything
echo "elm:/everything /everything nfs4 rw 0 0" >> etc/fstab

rm -f /boot/firstrun.sh /boot/firmware/firstrun.sh
sed -i 's| systemd.run.*||g' /boot/cmdline.txt /boot/firmware/cmdline.txt || true

exit 0
