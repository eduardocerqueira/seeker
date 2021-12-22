#date: 2021-12-22T17:11:04Z
#url: https://api.github.com/gists/e51573e1fd03b4493afe8681be9ddb0a
#owner: https://api.github.com/users/LinuxIntellect

# If you need to install playSMS on your ubuntu then contact with me:

# Skype: zobaer.ahmed5
# BiP: +8801818264577
# Whatsapp: +8801818 264577
# Telegram: +8801818 264577
# Viber: +8801818264577
# Signal: +8801818264577
# Email: linuxintellect@gmail.com
# https://www.linkedin.com/in/linuxintellect

# Youtube Video: 
############################################################################################################################

# Enable Ubuntu Firewall
# Allow SSH first:
sudo ufw allow ssh
sudo ufw enable
sudo ufw reload

## Upgrade Server
sudo apt update
sudo apt upgrade

sudo reboot
# Re-login SSH using user zobaer instead of root.

## Install MySQL Server
sudo apt install mariadb-server
sudo systemctl start mariadb.service
sudo systemctl enable mariadb.service
sudo mysql
exit;

## Install Web Server and PHP 7.2
# Install Apache2, PHP 7.2 and required PHP modules:
sudo apt install apache2 php php-cli php-mysql php-gd php-curl php-mbstring php-xml php-zip

# art Apache2 and enable it:
sudo systemctl start apache2.service
sudo systemctl enable apache2.service

# Allow HTTP and HTTPS:
sudo ufw allow http
sudo ufw allow https
sudo ufw reload

# Let’s test the PHP:
cd /var/www/html
sudo nano test.php

<?php
echo "Hello World";

playsmsd /home/zobaer/etc/playsmsd.conf start
playsmsd /home/zobaer/etc/playsmsd.conf status

# Go to your browser, browse the server and login as playSMS administrator, and change the default admin password immediately.
# Default admin access:

#   Username: admin
#   Password: admin
