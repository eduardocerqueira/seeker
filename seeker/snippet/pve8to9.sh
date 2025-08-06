#date: 2025-08-06T17:17:55Z
#url: https://api.github.com/gists/45b475fe17ac1b016f4ae30750a22d10
#owner: https://api.github.com/users/jershbytes

#!/usr/bin/env bash
# This script is used to upgrade Proxmox VE from version 8 to version 9.
# Author: Joshua Ross
# License: MIT License
# Version: 1.0
# Usage: Run this script as root or with sudo privileges.

set -eux

check_latest() {
	apt update
	apt dist-upgrade
	# Ensure pveversion is installed, or replace with a valid command
	command -v pveversion && pveversion || echo "pveversion not found"
}

# Update Debian Base Repositories to Trixie
update_debian_repos() {
    sed -i 's/bookworm/trixie/g' /etc/apt/sources.list
    sed -i 's/bookworm/trixie/g' /etc/apt/sources.list.d/pve-enterprise.list || true
}

# Add the Proxmox VE 9 Package Repository
add_pve9_repo() {
cat > /etc/apt/sources.list.d/proxmox.sources << EOF
Types: deb
URIs: http://download.proxmox.com/debian/pve
Suites: trixie
Components: pve-no-subscription
Signed-By: /usr/share/keyrings/proxmox-archive-keyring.gpg
EOF
}

# Update the Ceph Package Repository
add_pve9_ceph_repo() {
cat > /etc/apt/sources.list.d/ceph.sources << EOF
Types: deb
URIs: http://download.proxmox.com/debian/ceph-squid
Suites: trixie
Components: no-subscription
Signed-By: /usr/share/keyrings/proxmox-archive-keyring.gpg
EOF
}

# Upgrade the System
upgrade_system() {
    apt update
    apt dist-upgrade -y
}

# Main Upgrade Function
upgrade_pve() {
    check_latest
    update_debian_repos
    add_pve9_repo
    add_pve9_ceph_repo
    upgrade_system
}

# Execute the upgrade
upgrade_pve

# Clean up
apt autoremove -y
apt clean

# Final message
echo "Proxmox VE upgrade to version 9 completed successfully."
echo "Please reboot your system to complete the upgrade process."