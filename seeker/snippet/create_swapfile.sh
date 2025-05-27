#date: 2025-05-27T16:50:21Z
#url: https://api.github.com/gists/0d32887cbdb18be27bffe1ea7e2507ca
#owner: https://api.github.com/users/styczen

#!/usr/bin/env bash

# Usage: sudo ./create_swapfile.sh 50
# This script creates a swapfile of the size (in GB) given as the first argument.

set -e

if [[ $EUID -ne 0 ]]; then
  echo "Please run this script as root or with sudo."
  exit 1
fi

if [[ -z "$1" ]]; then
  echo "Usage: sudo $0 <size_in_GB>"
  exit 1
fi

SWAPFILE="/swapfile"
SIZE_GB="$1"

echo "Disabling any existing swap..."
swapoff -a

if [ -f "$SWAPFILE" ]; then
    echo "Removing existing swapfile..."
    rm -f "$SWAPFILE"
fi

echo "Creating a $SIZE_GB GB swapfile using dd..."
dd if=/dev/zero of=$SWAPFILE bs=1G count=$SIZE_GB status=progress

echo "Setting permissions on swapfile..."
chmod 600 $SWAPFILE

echo "Making swap area on the file..."
mkswap $SWAPFILE

echo "Enabling swapfile..."
swapon $SWAPFILE

FSTAB_ENTRY="$SWAPFILE none swap sw 0 0"
if ! grep -q "^$SWAPFILE " /etc/fstab; then
    echo "Adding swapfile to /etc/fstab for persistence..."
    echo "$FSTAB_ENTRY" >> /etc/fstab
else
    echo "Swapfile already present in /etc/fstab."
fi

echo "Swap setup complete. Current swap status:"
swapon --show
free -h
