#date: 2023-06-19T16:39:02Z
#url: https://api.github.com/gists/f2100bd384601a4c333da919be6918f2
#owner: https://api.github.com/users/samsesh

#!/bin/bash

# Prompt for necessary information
read -p "Enter S3 bucket name: " bucket_name
read -p "Enter the mount point directory path: " mount_point

# Provide example and get mount type
echo ""
echo "1. To add the mount to /etc/fstab"
echo ""
echo "2. To add the mount to crontab"
echo ""

read -p "Enter your choice (1 or 2): " choice

if [[ $choice == "1" ]]; then
    # Add to /etc/fstab
    read -p "Enter the username for fstab entry: " username
    echo "s3fs#${bucket_name} ${mount_point} fuse _netdev,allow_other 0 0" | sudo tee -a /etc/fstab
    echo "Added the following entry to /etc/fstab:"
    echo "s3fs#${bucket_name} ${mount_point} fuse _netdev,allow_other 0 0"
    echo "Please ensure that you have created the appropriate ~/.passwd-s3fs file with your AWS credentials."

elif [[ $choice == "2" ]]; then
    # Add to crontab
    crontab_entry="@reboot s3fs ${bucket_name} ${mount_point} -o passwd_file=~/.passwd-s3fs"
    (crontab -l 2>/dev/null; echo "$crontab_entry") | crontab -
    echo "Added the following entry to crontab:"
    echo "$crontab_entry"
    echo "Please ensure that you have created the appropriate ~/.passwd-s3fs file with your AWS credentials."

else
    echo "Invalid choice. Please enter either '1' or '2'."
    exit 1
fi
