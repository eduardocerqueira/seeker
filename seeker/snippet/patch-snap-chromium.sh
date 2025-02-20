#date: 2025-02-20T16:55:56Z
#url: https://api.github.com/gists/aed8773552d3b8201fe702da23ce7bc7
#owner: https://api.github.com/users/ermshiperete

#!/bin/bash
# Replace the ibus library (im-ibus.so) included in the gnome-42-2204 snap with
# a newer one that fixes surrounding text detection.
set -e

COLOR_RED='\x1b[31m'                # $(tput setaf 1)
COLOR_GREEN='\x1b[32m'              # $(tput setaf 2)
COLOR_BLUE='\x1b[34m'               # $(tput setaf 4)
COLOR_RESET='\x1b(B\x1b[m'          # $(tput sgr0)

if [[ "${USER}" != "root" ]]; then
  echo -e "${COLOR_RED}This script must be run as root. Restarting with sudo...${COLOR_RESET}"
  sudo "$0" "$@"
  exit $?
fi

# we're running as root now

check_dependency() {
  if ! dpkg -l "$1" &>/dev/null; then
    echo -e "${COLOR_GREEN}Installing ${1}...${COLOR_RESET}"
    apt-get update && apt-get install -y "${1}"
  fi
}

check_dependency squashfs-tools
check_dependency wget

for f in /var/lib/snapd/snaps/gnome-42-2204_*.snap; do
  echo -e ""
  echo -e "${COLOR_GREEN}Processing ${f}:${COLOR_RESET}"
  echo -e "${COLOR_BLUE}Extracting squashfs...${COLOR_RESET}"
  [ ! -f "${f}.bak" ] && mv "${f}" "${f}.bak"
  unsquashfs -f -d /tmp/squashfs-root "${f}.bak"
  # replace ibus files included in the snap with the newer ones. We have to take
  # one compiled for Jammy (which corresponds to the core22 base snap) so that
  # dependent libraries are compatible.
  IBUS_GTK3="/tmp/ibus-gtk3_1.5.28"
  if [ ! -d "${IBUS_GTK3}" ]; then
    echo -e "${COLOR_BLUE}Downloading new ibus-gtk3...${COLOR_RESET}"
    rm -f "${IBUS_GTK3}.deb"
    wget -O "${IBUS_GTK3}.deb" https://launchpad.net/~keymanapp/+archive/ubuntu/keyman-alpha/+files/ibus-gtk3_1.5.28-3sil2~jammy_amd64.deb
    dpkg-deb -R "${IBUS_GTK3}.deb" "${IBUS_GTK3}"
  fi
  cp "${IBUS_GTK3}/usr/lib/x86_64-linux-gnu/gtk-3.0/3.0.0/immodules/im-ibus.so" \
    /tmp/squashfs-root/usr/lib/x86_64-linux-gnu/gtk-3.0/3.0.0/immodules/

  echo -e "${COLOR_BLUE}Repacking squashfs...${COLOR_RESET}"
  rm -f "${f}"
  mksquashfs /tmp/squashfs-root "${f}"
done

echo -e "${COLOR_GREEN}Done${COLOR_RESET}"
rm -rf /tmp/squashfs-root

echo -e "${COLOR_RED}Please reboot before using the patched snaps${COLOR_RED}"
