#date: 2023-11-14T16:58:57Z
#url: https://api.github.com/gists/89c888cf7efba6d85cecef5bda8605b2
#owner: https://api.github.com/users/linusromland

#!/bin/bash

# First-time setup script for Debian server
# Run like this:
# curl -s https://gist.githubusercontent.com/linusromland/89c888cf7efba6d85cecef5bda8605b2/raw/19eefa8834fb38d4aee837508b751448a7429b33/setup.sh | bash 

# Function to display confirmation prompt
confirm() {
    while true; do
        read -rp "Do you wish to proceed with the setup? [Y/n]: " yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Bypass confirmation with -y flag
if [ "$1" != "-y" ]; then
    confirm
fi

# Check if the script is run as root
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Update and upgrade the system
echo "Updating and upgrading the system..."
apt-get update
apt-get upgrade -y

# Install sudo if it's not already installed
if ! command -v sudo &> /dev/null; then
    echo "Installing sudo..."
    apt-get update
    apt-get install -y sudo
fi

# Add user "linusromland" to the sudoers file if not already added
if ! grep -q "linusromland" /etc/sudoers; then
    echo "Adding 'linusromland' to the sudoers file..."
    echo "linusromland ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
fi

echo "Setup completed successfully!"
