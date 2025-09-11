#date: 2025-09-11T17:03:01Z
#url: https://api.github.com/gists/25bc133911b08c255b9d825bb610b27b
#owner: https://api.github.com/users/TheEpicMax007

#!/usr/bin/env bash
# xfce-xp.sh – Full XFCE Windows XP Rice Script for Arch Linux
# Run this on a clean XFCE setup

set -e

echo "[*] Updating system..."
sudo pacman -Syu --noconfirm

echo "[*] Installing required packages..."
sudo pacman -S --needed --noconfirm \
  xfce4 xfce4-goodies \
  git wget unzip \
  lxappearance \
  papirus-icon-theme \
  ttf-dejavu ttf-liberation ttf-ms-fonts

# === THEMES & ICONS ===
XP_THEMES="$HOME/.themes"
XP_ICONS="$HOME/.icons"

mkdir -p "$XP_THEMES" "$XP_ICONS"

echo "[*] Downloading Windows XP theme (Chicago95)..."
cd /tmp
rm -rf Chicago95
git clone https://github.com/Elbullazul/Chicago95.git
cp -r Chicago95/Theme/Chicago95 "$XP_THEMES/"
cp -r Chicago95/Icons/Chicago95 "$XP_ICONS/"
cp -r Chicago95/Cursors/* "$XP_ICONS/"

# === XFCE SETTINGS ===
echo "[*] Applying XFCE theme settings..."

# GTK + WM Theme
xfconf-query -c xsettings -p /Net/ThemeName -s "Chicago95" --create -t string -s "Chicago95"
xfconf-query -c xfwm4 -p /general/theme -s "Chicago95" --create -t string -s "Chicago95"

# Icons + Cursor
xfconf-query -c xsettings -p /Net/IconThemeName -s "Chicago95" --create -t string -s "Chicago95"
xfconf-query -c xsettings -p /Gtk/CursorThemeName -s "Chicago95" --create -t string -s "Chicago95"

# Font
xfconf-query -c xsettings -p /Gtk/FontName -s "Tahoma 9" --create -t string -s "Tahoma 9"

# === WALLPAPER ===
echo "[*] Setting XP Bliss wallpaper..."
mkdir -p ~/Pictures/wallpapers
wget -O ~/Pictures/wallpapers/xp-bliss.jpg \
  "https://upload.wikimedia.org/wikipedia/en/2/26/Bliss_%28Windows_XP%29.png"
xfconf-query -c xfce4-desktop \
  -p /backdrop/screen0/monitor0/image-path \
  -s "$HOME/Pictures/wallpapers/xp-bliss.jpg"

# === PANEL CONFIG ===
echo "[*] Resetting XFCE panel to XP layout..."
xfconf-query -c xfce4-panel -p /panels -r -R
xfconf-query -c xfce4-panel -p /plugins -r -R
xfce4-panel --quit || true
pkill xfconfd || true

sleep 2

# Create panel
xfconf-query -c xfce4-panel -p /panels -t int -s 1 --create
xfconf-query -c xfce4-panel -p /panels/panel-1/position -t string -s "p=6;x=0;y=0" --create
xfconf-query -c xfce4-panel -p /panels/panel-1/size -t int -s 28 --create
xfconf-query -c xfce4-panel -p /panels/panel-1/length -t int -s 100 --create

# Add plugins: whiskermenu, tasklist, systray, clock
xfconf-query -c xfce4-panel -p /plugins/plugin-1 -t string -s "whiskermenu" --create
xfconf-query -c xfce4-panel -p /plugins/plugin-2 -t string -s "tasklist" --create
xfconf-query -c xfce4-panel -p /plugins/plugin-3 -t string -s "separator" --create
xfconf-query -c xfce4-panel -p /plugins/plugin-4 -t string -s "systray" --create
xfconf-query -c xfce4-panel -p /plugins/plugin-5 -t string -s "clock" --create

xfconf-query -c xfce4-panel -p /panels/panel-1/plugin-ids -t int -t int -t int -t int -t int \
  -s 1 -s 2 -s 3 -s 4 -s 5 --create

# Restart panel
xfce4-panel &

echo
echo "[+] Windows XP rice applied!"
echo "    → Theme: Chicago95"
echo "    → Icons: Chicago95"
echo "    → Cursors: Chicago95"
echo "    → Font: Tahoma 9"
echo "    → Wallpaper: XP Bliss"
echo "    → Panel: Whisker Menu + Tasklist + Tray + Clock"
echo
echo ">>> Log out/in or restart panel if something looks off."
