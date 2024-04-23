#date: 2024-04-23T16:53:06Z
#url: https://api.github.com/gists/f354cdb70ba5c1973e03884bd36e4f4f
#owner: https://api.github.com/users/JSONOrona

#!/bin/bash

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root" 1>&2
  exit 1
fi

# Check for required argument
if [ $# -ne 2 ]; then
  echo "Usage: "**********"
  exit 1
fi

USERNAME=$1
PASSWORD= "**********"

# Check if the user already exists
if id "$USERNAME" &>/dev/null; then
  echo "User $USERNAME already exists."
  exit 1
fi

# Add the user with a home directory
adduser --disabled-password --gecos "" "$USERNAME"

if [ $? -ne 0 ]; then
  echo "Failed to add user $USERNAME."
  exit 1
fi

# Set the user's password
echo "$USERNAME: "**********"

if [ $? -ne 0 ]; then
  echo "Failed to set password for user $USERNAME."
  userdel "$USERNAME"
  exit 1
fi

# Add user to the sudo group
usermod -aG sudo "$USERNAME"

if [ $? -ne 0 ]; then
  echo "Failed to add user $USERNAME to sudo group."
  userdel "$USERNAME"
  exit 1
fi

echo "User $USERNAME added successfully with sudo privileges."
vileges."
