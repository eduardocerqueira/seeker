#date: 2023-10-02T16:53:56Z
#url: https://api.github.com/gists/5241b334a0b97c4984d0b7dcde833592
#owner: https://api.github.com/users/ThinGuy

#!/bin/bash

##############################
# Scripted XRDP Installation #
##############################

#
# Get Ubuntu release
#
export UREL=$(lsb_release 2>/dev/null -sr|sed 's/\.//g')

#
# Base package list (Jammy) with Lunar and Mantic shell additions.
#
declare -ag PKGLIST=(xserver-xorg-input-all xrdp gnome-tweaks gnome-shell gnome-shell-common gnome-shell-extensions gnome-shell-extension-appindicator gnome-shell-extension-desktop-icons-ng gnome-shell-extension-manager gnome-shell-extension-prefs gnome-shell-extension-ubuntu-dock)
[[ ${UREL} -ge 2304 ]] && { PKGLIST+=("gnome-shell-extension-ubuntu-tiling-assistant"); }
[[ ${UREL} -ge 2310 ]] && { PKGLIST+=("gnome-shell-ubuntu-extensions"); }

#
# Install xrdp and gnome-shell components from PKGLIST
#
sudo apt install -yqf --auto-remove --purge ${PKGLIST[@]}

#
# Enable User-Themes in Gnome-Shell
#
gnome-extensions enable user-theme@gnome-shell-extensions.gcampax.github.com

#
# Set Gnome-Shell User-Theme to based on current light/dark prefs
#
export DEF_THEME="Yaru"
[[ ${UREL} -lt 2304 ]] && { gsettings set org.gnome.desktop.interface gtk-theme "'${DEF_THEME}-dark'"; }
[[ ${UREL} -lt 2304 ]] && { gsettings set org.gnome.desktop.interface gtk-theme "'${DEF_THEME}'"; }
[[ ${UREL} -ge 2304 && $(gsettings get org.gnome.desktop.interface color-scheme) =~ dark ]] && { echo dconf write /org/gnome/shell/extensions/user-theme/name "'${DEF_THEME}-dark'"; }
[[ ${UREL} -ge 2304 && $(gsettings get org.gnome.desktop.interface color-scheme) =~ default ]] && { dconf write /org/gnome/shell/extensions/user-theme/name "'${DEF_THEME}'"; }

#
# Ensure all users can connect to remote desktop
#
sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

#
# Help RDP XDG paths and style names
#
if [[ ! $(grep -q 'XDG_CURRENT_DESKTOP=ubuntu:GNOME' /etc/xrdp/startwm.sh;echo $?) ]];then
  sudo sed -i "4 a cat <<EOF > ~/.xsessionrc\nexport GNOME_SHELL_SESSION_MODE=ubuntu\nexport XDG_CURRENT_DESKTOP=ubuntu:GNOME\nexport XDG_CONFIG_DIRS=/etc/xdg/xdg-ubuntu:/etc/xdg\nEOF\n" /etc/xrdp/startwm.sh
fi

#
# Set polkit dir
#
[[ -d /etc/polkit-1/localauthority/50-local.d ]] && { export PKDIR="/etc/polkit-1/localauthority/50-local.d"; }
[[ -d /etc/polkit-1/localauthority.conf.d ]] && { export PKDIR="/etc/polkit-1/localauthority.conf.d"; }


#
# Fix color profile error
#
cat <<EO1 |sudo tee 1>/dev/null ${PKDIR}/45-allow.colord.pkla
# Fix "authentication is required to create a color profile" error
[Allow Colord all Users]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
ResultAny=no
ResultInactive=no
ResultActive=yes
EO1

#
# Fix issues installing software from app center via RDP
#
cat <<EO2 |sudo tee 1>/dev/null ${PKDIR}/46-allow.pkg-mgmt.pkla
# Fix "You do not have permission" when installing from app center (aka Ubuntu Software, aka software center)
[Allow Package Management all Users]
Identity=unix-user:*
Action=org.debian.apt.*;io.snapcraft.*;org.freedesktop.packagekit.*;com.ubuntu.update-notifier.*
ResultAny=no
ResultInactive=no
ResultActive=yes
EO2

#
# Fix system management tasks via RDP
#
cat <<EO3 |sudo tee 1>/dev/null ${PKDIR}/46-allow.remote-admin.pkla
[Allow Remote Admin]
Identity=unix-group:*
Action=*
ResultAny=auth_admin_keep
ResultInactive=auth_admin_keep
ResultActive=auth_admin_keep
EO3


#
# Restart xrdp and policy-kit services
#
sudo systemctl restart xrdp-sesman xrdp polkit