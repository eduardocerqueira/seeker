#date: 2022-03-01T16:54:04Z
#url: https://api.github.com/gists/531ccf365dd84353471c6a7a99549144
#owner: https://api.github.com/users/mrl22

#!/bin/bash
# ------------------------------------------------------------------
# Am i Root user?
if [ $(id -u) -eq 0 ]; then
        read -p "Enter username : " username
        read -s -p "Enter password : " password
        egrep "^$username" /etc/passwd >/dev/null
        if [ $? -eq 0 ]; then
                echo "$username exists!"
                exit 1
        else
                pass=$(perl -e 'print crypt($ARGV[0], "password")' $password)
                useradd -s /bin/bash -g backup -m -p "$pass" "$username"
                [ $? -eq 0 ] && echo "User has been added to system!" || echo "Failed to add a user!"
                chmod 700 /home/$username
        fi
else
        echo "Only root may add a user to the system."
        exit 2
fi