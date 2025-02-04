#date: 2025-02-04T16:39:24Z
#url: https://api.github.com/gists/d9cacda7093f6da5fc8029498ac85a93
#owner: https://api.github.com/users/ve7mjc

#!/bin/bash

# Ansible Provisioner Host Init script
# BCWARN SDR Hub
# Matt Currie VE7MJC 2025-02-04
#
# - exit if user exists
# - install `sudo` if not-exist
# - create user
# - create `$user/.ssh/authorized_keys`
# - insert $SSH_PUBKEY into `$user/.ssh/authorized_keys`
# - add $USER to sudoers with NOPASSWD
# - retrieve and set user ssh pubkey if cataloged
# - profit.
#

USER="provisioner"
SSH_PUBKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCxldXJZnACEGdu54OzxA8ddAqIDhKPmm4CwXf4TZZ18T3xNRY06lv8Ke4CDCc6/4ZFJq4jY4ExjCIQfGUXx4Tja52NvueOTMMsA9EwkmSGmzP4jxHFR0GBbcOcYnxf93XoJjl8wY3YpbhFWxeEzHjp2J5ntXYEIb/DwoJG/rdNwew/xzRgK6gci/jTsZH8FWB1hnuxTLww7CAb/DJ1rlWcnI0X+3Ec7TkalxGONEeOReiSfGDI4rNpOX+QJ+cJgYkKHrtBVsYTOcN9fDwLFCEN7RDjiQeeYn/ZTXKKgEdx15gQVYWHGZPrtyA/D6Y6V9/PO+JoRJgl7ocfWXovMlaSSHrk3fw078B4AVzJNt9FZBEjed2lKrSufKffHr+L6YtHLJFMZQZvC7iLoGVN16ck6pLuiunrXP+FoFEpCQfCYj4Rlrg+yfAGiyC+xh356tD2ZqUYeHbZZzx894Twoy3gkVZH8kmg7QGvFPz1E9LWYTHnapfulQvKUhB47JTUj8By17aVZud+SDk5Dm5WO/uHjXhx27eM6xAaf0fIEsAOD9xBkrOogq2PiqTGhwr3br0WgqjQsedb8uW/8ikuwLb3c0EuUNewUaWaCzQsWDuLMpeLHob7/w1RXFzqrzDT6WvcttkjIMj3+SEtQH+adk23qzu0ifMmFmVYc3miu3dUVQ== provisioner@sdrhub"

AUTHORIZED_KEYS_FILE="/home/${USER}/.ssh/authorized_keys"

# Ensure script is run as root
if [[ "$(id -u)" -ne 0 ]]; then
    echo "Error: This script must be run as root."
    exit 1
fi

# Check if user already exists
if id "$USER" &>/dev/null; then
    echo "Error: User '$USER' already exists."
    exit 1
fi

# Check if sudo is installed, install if missing
if ! command -v sudo &> /dev/null; then
    echo "- Sudo not found. Installing..."
    apt update && apt install -y sudo
fi

# Create the user without a password
echo "- Creating user: $USER"
useradd -m -s /bin/bash "$USER"

# Create .ssh directory if not exists
SSH_DIR="/home/$USER/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"
chown "$USER:$USER" "$SSH_DIR"

# Check if authorized_keys exists
if [ ! -f "$AUTHORIZED_KEYS_FILE" ]; then
    echo "- Creating authorized_keys ('$AUTHORIZED_KEYS_FILE')"
    sudo mkdir -p "/home/$USER/.ssh"
    sudo chmod 700 "/home/$USER/.ssh"
    sudo touch "$AUTHORIZED_KEYS_FILE"
    sudo chmod 600 "$AUTHORIZED_KEYS_FILE"
    sudo chown "$USER:$USER" "$AUTHORIZED_KEYS_FILE"
fi

# Add SSH public key if not already present
AUTHORIZED_KEYS="$SSH_DIR/authorized_keys"
if ! grep -qxF "$SSH_PUBKEY" "$AUTHORIZED_KEYS" 2>/dev/null; then
    echo "$SSH_PUBKEY" >> "$AUTHORIZED_KEYS"
    chmod 600 "$AUTHORIZED_KEYS"
    chown "$USER:$USER" "$AUTHORIZED_KEYS"
    echo "- Added SSH public key to $AUTHORIZED_KEYS"
else
    echo "- Warning: SSH public key already exists in $AUTHORIZED_KEYS"
fi

# Add user to sudo group
usermod -aG sudo "$USER"

# Configure sudo access without password
SUDO_FILE="/etc/sudoers.d/$USER"
echo "$USER ALL=(ALL) NOPASSWD:ALL" > "$SUDO_FILE"
chmod 440 "$SUDO_FILE"
echo "- Added $USER to sudo + NOPASSWD"

echo "- User '$USER' configured successfully."
exit 0
