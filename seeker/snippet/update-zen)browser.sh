#date: 2024-11-12T16:50:52Z
#url: https://api.github.com/gists/71aa425ef6d0386e0f1f1c00ad045c4a
#owner: https://api.github.com/users/nstCactus

#!/usr/bin/env bash

set -e

archiveFilename="/home/ybianchi/Downloads/zen.linux-specific.tar.bz2"

if [ "$EUID" -ne 0 ]; then
  # Print an error message in red
  echo -e "\e[31mError: This script must be run as root.\e[0m"
  exit 1
fi

if [[ ! -f "$archiveFilename" ]]; then
  echo -e "\e[31mError: $archiveFilename does not exist. Please download Zen browser before running this script.\e[0m"
  exit 1
fi

echo "Starting Zen browser update…"
echo -e "  \e[33m⮡ Finish what you were doing and press any key to close Zen browser or Ctrl+C to abort.\e[0m"
read -n 1 -s

killall zen || true

echo -n '  ⮡ Removing the previous zen version…'
rm -rf /opt/zen
echo -e " \e[32mdone\e[0m"

cd /opt

echo -n '  ⮡ Extracting new version to /opt/zen…'
tar xf "$archiveFilename"
echo -e " \e[32mdone\e[0m"

echo -n '  ⮡ Cleaning up…'
rm /home/ybianchi/Downloads/zen.linux-specific.tar.bz2
echo -e " \e[32mdone\e[0m"

echo -e '\e[34mUpdate sucessfull\e[0m'
