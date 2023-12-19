#date: 2023-12-19T17:08:27Z
#url: https://api.github.com/gists/800635549fcbb4c83cce535647a347aa
#owner: https://api.github.com/users/selimyasar

#!/bin/bash

# prepare a Ubuntu server minimal installation with working network using network manager
# this works on raspbian, too

apt-get install software-properties-common -y

# install KODI
add-apt-repository ppa:team-xbmc/ppa -y
apt-get update && apt-get install -y kodi xserver-xorg xinit dbus-x11 alsa-utils avahi-daemon 

# allow poweroff for everyone
echo -e "allowed_users=anybody\nneeds_root_rights=yes" >/etc/X11/Xwrapper.config
cat <<EOF >/etc/polkit-1/localauthority/50-local.d/all_users_shutdown_reboot.pkla
[Allow all users to shutdown and reboot]
Identity=unix-user:*
Action=org.freedesktop.login1.*;org.freedesktop.upower.*;org.freedesktop.consolekit.system.*
ResultActive=yes
ResultAny=yes
ResultInactive=yes
EOF

# add kodi user
KHOME=/var/lib/kodi/
adduser --system --group --home $KHOME --disabled-password kodi
adduser kodi audio
adduser kodi video

# init script
cat <<EOF >$KHOME/.xinitrc
#!/bin/bash
dbus-launch --exit-with-session &
kodi-standalone
EOF

# kodi systemd service
cat <<EOF >/etc/systemd/system/kodi.service
[Unit]
Description = Kodi Media Center
After = systemd-user-sessions.service network-online.target sound.target

[Service]
User = kodi
Group = kodi
ExecStart = /usr/bin/xinit -- :0 -nolisten tcp vt7
Restart = always
RestartSec = 5

[Install]
WantedBy = multi-user.target
EOF

systemctl daemon-reload && systemctl enable kodi

# create a hotspot when no network connection can be set up
# create the hotspot once with 
# nmcli device wifi hotspot ssid foobar password foobar123
cat <<EOF >/etc/systemd/system/hotspot.service
[Unit]
Description = On-Demand WiFi Hotspot
After = network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre = /bin/sh -c '! nm-online -q'
ExecStart = /usr/bin/nmcli connection up Hotspot
ExecStop = /usr/bin/nmcli connection down Hotspot

[Install]
WantedBy = multi-user.target
EOF

systemctl daemon-reload && systemctl enable hotspot

exit

# create wireless presenter by using vncviewer in listen mode
apt-get install -y unclutter xdotool xloadimage xvnc4viewer xterm x11-xserver-utils
apt-get install -y --no-install-recommends openbox

cat <<EOF >$KHOME/.xinitrc
#!/bin/bash
cd \$HOME
dbus-launch --exit-with-session &
xset -dpms; xset s off
test -f background.* && xsetbg -center background.* 
unclutter -root &
xvncviewer -listen -fullscreen &
openbox

#SOUND="--alsa-output-device=plughw:CARD=Device,DEV=0"
#SIZE=\$(xrandr |grep \* |awk '{print \$1}')
#WINDOW="--window-position=0,0 --window-size=\${SIZE/x/,}"
#OPTS="--start-maximized --disable-translate --disable-new-tab-first-run --no-default-browser-check --no-first-run"
#exec google-chrome \$OPTS \$WINDOW \$SOUND "\$@"
EOF

# optionally install chrome 
#wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub |apt-key add - 
#sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
#apt-get update && apt-get install -y google-chrome-stable 

apt-get install -y --no-install-recommends chromium-browser
