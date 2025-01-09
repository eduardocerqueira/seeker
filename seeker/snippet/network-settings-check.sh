#date: 2025-01-09T17:12:09Z
#url: https://api.github.com/gists/b605df329647c470fae995b68b0b1e7c
#owner: https://api.github.com/users/nickkostov

#!/bin/bash

# Function to check distribution type
check_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    else
        echo "Unable to determine the distribution type."
        exit 1
    fi
}

# Main script
check_distro

# Print network configuration
echo "\n####################### Network Configuration #######################"
ifconfig 2>/dev/null || ip addr

echo "\n######################### /etc/hosts File ##########################"
cat /etc/hosts

echo "\n####################### /etc/resolv.conf File ######################"
cat /etc/resolv.conf

# Handle systemd-resolved files
echo "\n####################### Systemd Resolved Config #####################"
if [ "$DISTRO" == "debian" ] || [ "$DISTRO" == "ubuntu" ]; then
    if [ -f /etc/systemd/resolved.conf ]; then
        echo "\n/etc/systemd/resolved.conf:" 
        cat /etc/systemd/resolved.conf
    else
        echo "File not found: /etc/systemd/resolved.conf"
    fi

    if [ -f /run/systemd/resolve/resolv.conf ]; then
        echo "\n/run/systemd/resolve/resolv.conf:" 
        cat /run/systemd/resolve/resolv.conf
    else
        echo "File not found: /run/systemd/resolve/resolv.conf"
    fi
elif [ "$DISTRO" == "rhel" ] || [ "$DISTRO" == "centos" ] || [ "$DISTRO" == "fedora" ]; then
    if [ -f /etc/systemd/resolved.conf ]; then
        echo "\n/etc/systemd/resolved.conf:" 
        cat /etc/systemd/resolved.conf
    else
        echo "File not found: /etc/systemd/resolved.conf"
    fi
else
    echo "Unsupported distribution: $DISTRO"
    exit 1
fi

echo "\n#####################################################################"
