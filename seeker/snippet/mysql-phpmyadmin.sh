#date: 2022-12-26T16:24:15Z
#url: https://api.github.com/gists/a8eec8fbc399643f609b11a04924393a
#owner: https://api.github.com/users/base2code

#!/bin/bash

echo "------------------------"
echo "Updating package lists"
echo "------------------------"
sleep 2
apt-get update

echo "------------------------"
echo "Installing MariaDB"
echo "------------------------"
sleep 2
apt-get install mariadb-server -y

echo "------------------------"
echo "Installing Apache2 & PHP"
echo "------------------------"
sleep 2
apt-get install apache2 php7.4 -y

echo "------------------------"
echo "Starting MariaDB"
echo "------------------------"
sleep 2
service mariadb start
systemctl enable mariadb

echo "------------------------"
echo "Securing MySQL Installation"
echo "------------------------"
mysql_secure_installation

echo "------------------------"
echo "Installing phpmyadmin"
echo "------------------------"
sleep 2
apt-get install phpmyadmin -y

echo "------------------------"
echo "Restarting apache2"
echo "------------------------"
service apache2 restart

echo "------------------------"
echo "Script finished!"
echo "------------------------"