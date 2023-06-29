#date: 2023-06-29T17:06:59Z
#url: https://api.github.com/gists/cf7d66bd151afcd67875b0d930278a6c
#owner: https://api.github.com/users/benediktms

#!/bin/bash

# Get a list of all 'nixbld' users
users=$(getent passwd | grep nixbld | cut -d: -f1)

# Loop over the users and delete each one
for user in $users; do
    echo "Deleting user: $user"
    sudo userdel -r "$user"
done

# Delete 'nixbld' group
echo "Deleting group: nixbld"
sudo groupdel nixbld
