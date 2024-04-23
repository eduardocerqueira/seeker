#date: 2024-04-23T16:59:15Z
#url: https://api.github.com/gists/87838121856fd5621955a7af4f186a26
#owner: https://api.github.com/users/JSONOrona

#!/bin/bash

# Ensures the script is run with root privileges
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root" 1>&2
  exit 1
fi

# Checks for the required argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <new_hostname>"
  exit 1
fi

NEW_HOSTNAME=$1

# Sets the hostname by updating /etc/hostname
echo "$NEW_HOSTNAME" > /etc/hostname

# Updates /etc/hosts to reflect the change and preserve system integrity
sed -i "s/127\.0\.1\.1\s.*/127.0.1.1\t$NEW_HOSTNAME/" /etc/hosts

# Applies the hostname change without needing a restart
hostnamectl set-hostname "$NEW_HOSTNAME"

echo "Hostname changed successfully to $NEW_HOSTNAME"
