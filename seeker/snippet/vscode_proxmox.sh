#date: 2024-12-27T17:04:12Z
#url: https://api.github.com/gists/17fcea15aeefa5fc472ad7228c0a5957
#owner: https://api.github.com/users/Battiatus

#!/usr/bin/env bash

# Copyright (c) 2024 ayaros-technology
# Author: kronos

# Function to display header
function header_info {
    cat <<"EOF"
   ______          __        _____                          
  / ____/___  ____/ /__     / ___/___  ______   _____  _____
 / /   / __ \/ __  / _ \    \__ \/ _ \/ ___/ | / / _ \/ ___/
/ /___/ /_/ / /_/ /  __/   ___/ /  __/ /   | |/ /  __/ /    
\____/\____/\__,_/\___/   /____/\___/_/    |___/\___/_/     
 
EOF
}

# Variables
IP=$(hostname -I | awk '{print $1}')
YW="\033[33m"
BL="\033[36m"
RD="\033[01;31m"
GN="\033[1;92m"
CL="\033[m"
APP="Code Server"
hostname="$(hostname)"

# Exit script on error
set -o errexit
set -o errtrace
set -o nounset
set -o pipefail

# Error handling
trap 'echo -e "${RD}‼ ERROR ${CL}$?@$LINENO" >&2; exit 1' ERR

# Clear screen and display header
clear
header_info

# Environment checks
if command -v pveversion >/dev/null 2>&1; then
    echo -e "⚠️  Can't Install on Proxmox"
    exit 1
fi

if [ -e /etc/alpine-release ]; then
    echo -e "⚠️  Can't Install on Alpine"
    exit 1
fi

# User confirmation
while true; do
    read -p "This will Install ${APP} on $hostname. Proceed (y/n)? " yn
    case $yn in
        [Yy]*) break ;;
        [Nn]*) exit ;;
        *) echo "Please answer yes or no." ;;
    esac
done

# Functions for status messages
function msg_info() {
    echo -ne "${YW}$1...${CL}"
}

function msg_ok() {
    echo -e "${GN}$1${CL}"
}

# Install dependencies
msg_info "Installing Dependencies"
apt-get update
apt-get install -y curl git
msg_ok "Installed Dependencies"

# Verify curl exists
if ! command -v curl &>/dev/null; then
    echo -e "${RD}Curl is required but not installed. Exiting.${CL}"
    exit 1
fi

# Get latest version of code-server
VERSION=$(curl -s https://api.github.com/repos/coder/code-server/releases/latest |
    grep "tag_name" | awk -F\" '{print $4}' | sed 's/v//')

msg_info "Installing Code-Server v${VERSION}"
curl -fOL "https://github.com/coder/code-server/releases/tag/v${VERSION}/code-server_${VERSION}_amd64.deb"
dpkg -i "code-server_${VERSION}_amd64.deb"
rm -f "code-server_${VERSION}_amd64.deb"
msg_ok "Installed Code-Server v${VERSION}"

# Configure code-server
mkdir -p ~/.config/code-server/
cat <<EOF >~/.config/code-server/config.yaml
bind-addr: 0.0.0.0:8680
auth: none
password: "**********"
cert: false
EOF

# Enable and restart code-server
systemctl enable --now code-server@$USER
systemctl restart code-server@$USER
msg_ok "Configured Code-Server on $hostname"

# Final message
echo -e "${APP} is accessible at: ${BL}http://$IP:8680${CL}\n"
