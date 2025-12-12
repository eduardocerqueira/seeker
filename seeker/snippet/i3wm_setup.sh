#date: 2025-12-12T17:12:15Z
#url: https://api.github.com/gists/73475ea0bb28eb9bd33721c6f89484aa
#owner: https://api.github.com/users/amaan028-sudo

#!/bin/bash

------------------------------

1. Update System

------------------------------

sudo apt update && sudo apt upgrade -y

------------------------------

2. Core Utilities

------------------------------

sudo apt install -y sudo git curl wget unzip xorg

------------------------------

3. Window Manager + Display Manager

------------------------------

sudo apt install -y i3 i3status i3blocks lightdm lightdm-gtk-greeter

------------------------------

4. Terminal & File Manager

------------------------------

sudo apt install -y alacritty thunar thunar-archive-plugin

------------------------------

5. Browser & Development

------------------------------

sudo apt install -y firefox-esr code

------------------------------

6. Utilities & Apps

------------------------------

sudo apt install -y gnome-disk-utility gnome-calculator gnome-text-editor ristretto vlc
sudo apt install -y flameshot brightnessctl

------------------------------

7. Audio

------------------------------

sudo apt install -y pipewire pipewire-pulse pipewire-audio wireplumber pavucontrol

------------------------------

8. Network & Bluetooth

------------------------------

sudo apt install -y network-manager network-manager-gnome
sudo systemctl enable --now NetworkManager

sudo apt install -y bluetooth bluez blueman
sudo systemctl enable --now bluetooth

------------------------------

9. Launcher

------------------------------

sudo apt install -y rofi

------------------------------

10. Performance & Power

------------------------------

sudo apt install -y tlp tlp-rdw
sudo systemctl enable --now tlp
sudo systemctl enable --now fstrim.timer

------------------------------

11. Disable Unneeded Services

------------------------------

sudo systemctl disable --now avahi-daemon
sudo systemctl disable --now cups
sudo systemctl disable --now ModemManager

------------------------------

12. Touchpad Configuration

------------------------------

sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/X11/xorg.conf.d/40-libinput.conf > /dev/null <<EOL
Section "InputClass"
Identifier "touchpad"
MatchIsTouchpad "on"
Driver "libinput"
Option "Tapping" "on"
Option "NaturalScrolling" "true"
Option "ScrollMethod" "edge"
Option "DisableWhileTyping" "true"
EndSection
EOL

------------------------------

14. Cleanup

------------------------------

sudo apt autoremove --purge -y
sudo apt autoclean -y
sudo apt clean -y

echo "✅ i3WM installation complete! Reboot to start using i3."