#date: 2024-04-23T17:08:48Z
#url: https://api.github.com/gists/895af6dc49cbc1d60e27e54b4a8d2819
#owner: https://api.github.com/users/mik0l

# compact view
xfconf-query -c thunar -p /last-view -s ThunarCompactView

# show hidden files
xfconf-query -c thunar -p /last-show-hidden -nt bool -s true

# disable thumbnails
xfconf-query -c thunar -p /misc-thumbnail-mode -nt string -s THUNAR_THUMBNAIL_MODE_NEVER

# system inactivity
xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/inactivity-on-ac -nt uint -s 60

# disable effects
xfconf-query -c xfwm4 -p /general/use_compositing -s false

# disable unnecessary applications
mkdir -p ~/.config/autostart
echo -e "[Desktop Entry]\nHidden=true" | tee \
  ~/.config/autostart/blueman.desktop \
  ~/.config/autostart/geoclue-demo-agent.desktop \
  ~/.config/autostart/org.freedesktop.problems.applet.desktop \
  ~/.config/autostart/org.mageia.dnfdragora-updater.desktop \
  ~/.config/autostart/tracker-miner-fs-3.desktop \
  ~/.config/autostart/tracker-miner-rss-3.desktop
