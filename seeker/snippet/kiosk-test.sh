#date: 2024-04-23T16:50:18Z
#url: https://api.github.com/gists/ab026f46d552b44d84ea93ac535e1ed9
#owner: https://api.github.com/users/JSONOrona

#!/bin/bash

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

# Variables
USER_NAME="your_username"  # CHANGE this to the username you wish to auto-login

# Enable auto-login
echo "[Seat:*]
autologin-user=$USER_NAME
autologin-user-timeout=0" > /etc/lightdm/lightdm.conf

# Remove all panels
xfconf-query -c xfce4-panel -p /panels -rR

# Disable desktop icons
xfconf-query -c xfce4-desktop -p /desktop-icons/style -s 0

# Setup kiosk mode
mkdir -p /etc/xdg/xfce4/kiosk
echo "[xfce4-panel]
CustomizePanel=NONE

[xfce4-desktop]
CustomizeDesktop=NONE" > /etc/xdg/xfce4/kiosk/kioskrc

# Setup hotkey for terminal (Ctrl+Alt+T)
mkdir -p /etc/xdg/xfce4/xfconf/xfce-perchannel-xml
echo '<?xml version="1.0" encoding="UTF-8"?>

<channel name="xfce4-keyboard-shortcuts" version="1.0">
  <property name="commands" type="empty">
    <property name="custom" type="empty">
      <property name="&lt;Primary&gt;&lt;Alt&gt;t" type="string" value="xfce4-terminal"/>
    </property>
  </property>
</channel>' > /etc/xdg/xfce4/xfconf/xfce-perchannel-xml/xfce4-keyboard-shortcuts.xml

# Reload LightDM to apply changes
systemctl restart lightdm

echo "Setup complete. System will now log in automatically to a minimal desktop. Use Ctrl+Alt+T to open the terminal."
