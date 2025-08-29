#date: 2025-08-29T16:50:56Z
#url: https://api.github.com/gists/9299f7d60d2061853598af91219c62af
#owner: https://api.github.com/users/peppapig450

#!/usr/bin/env bash
set -Eeuo pipefail

apt purge ---auto-remove cmake
apt install -y software-properties-common lsb-release

wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

apt update 
apt install -y kitware-archive-keyring
apt install -y cmake
