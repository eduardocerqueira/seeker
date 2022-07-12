#date: 2022-07-12T16:52:47Z
#url: https://api.github.com/gists/027981924776c183ce3b83efcd69f6c7
#owner: https://api.github.com/users/jgalec

#!/bin/bash

red='\033[0;31m'
green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
magenta='\033[0;35m'
cyan='\033[0;36m'
# Clear the color after that
reset='\033[0m'

# Parámetros para dnf.conf
# https://www.dnf.io/docs/en/dnf-conf.html
echo -e "${yellow}Configurando dnf.conf${reset}"
echo "defaultyes=True" >> /etc/dnf/dnf.conf

# RPM Fusion
echo -e "${yellow}Instalando RPM Fusion${reset}"
dnf install -y https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
echo -e "${green}RPM Fusion instalado${reset}"

dnf upgrade --refresh
echo -e "${green}Repositorios actualizados${reset}"

# Programas básicos
echo -e "${yellow}Instalando programas básicos${reset}"
dnf install -y kernel-headers kernel-devel wget git git-lfs bash-completion xdg-{utils,user-dirs} neofetch tree
echo -e "${green}Programas básicos instalados${reset}"

# Fuentes
echo -e "${yellow}Instalando fuentes${reset}"
dnf install -y google-noto-cjk-fonts google-noto-emoji-fonts
echo -e "${green}Fuentes instaladas${reset}"

# Códecs multimedia
echo -e "${yellow}Instalando codecs multimedia${reset}"
dnf install -y gstreamer1-plugins-{bad-\*,good-\*,base} gstreamer1-plugin-openh264 gstreamer1-libav --exclude=gstreamer1-plugins-bad-free-devel
dnf install -y lame\* --exclude=lame-devel
dnf group upgrade --with-optional Multimedia
echo -e "${green}Codecs multimedia instalados${reset}"

# Flatpak
echo -e "${yellow}Instalando Flatpak${reset}"
dnf install -y flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
echo -e "${green}Flatpak instalado${reset}"

# Xorg
echo -e "${yellow}Instalando Xorg${reset}"
dnf install -y @base-x
echo -e "${green}Xorg instalado${reset}"

# Cinnamon
excluded_packages_list=("gnome-calendar"
                        "gnome-calculator"
                        "gnome-disk-utility"
                        "gnome-screenshot"
                        "gnome-system-monitor"
                        "gnome-terminal"
                        "hexchat"
                        "paper-icon-theme"
                        "pidgin"
                        "powerline"
                        "powerline-fonts"
                        "redshift-gtk"
                        "setroubleshoot"
                        "shotwell"
                        "simple-scan"
                        "thunderbird"
                        "tmux"
                        "tmux-powerline"
                        "transmission"
                        "vim-powerline"
                        "xawtv"
                        "xfburn")

printf -v excluded_packages '%s,' "${excluded_packages_list[@]}"

echo -e "${yellow}Instalando Cinnamon${reset}"
dnf group install "Cinnamon" --exclude="${excluded_packages%,}"
echo -e "${green}Cinnamon instalado${reset}"

# Reboot after 10 seconds
echo -e "${yellow}Reiniciando en 10 segundos${reset}"
sleep 10
reboot