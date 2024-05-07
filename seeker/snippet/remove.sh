#date: 2024-05-07T16:56:44Z
#url: https://api.github.com/gists/7566c6ea4715542db1aa05e3e0168de3
#owner: https://api.github.com/users/samuelnarioSOREVAN

# Stop the kiosk service
sudo systemctl stop kiosk

# Remove the kiosk service from startup
sudo systemctl disable kiosk

# Remove the kiosk service
sudo rm -f /etc/systemd/system/kiosk.service

# Reload systemctl daemons
sudo systemctl daemon-reload

# Remove the kiosk startup file and directory
sudo rm -rf /opt/kiosk

# Remove X.org, OpenBox, Firefox and autoremove associated programs no longer in use
sudo apt remove --autoremove -y xorg openbox firefox
