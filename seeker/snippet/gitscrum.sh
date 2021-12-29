#date: 2021-12-29T17:16:23Z
#url: https://api.github.com/gists/efb1756066c4c0c96eac6af33af54700
#owner: https://api.github.com/users/LinuxIntellect

# If you need to install Gitscrum: Agile Project Management tool on your ubuntu then contact with me:

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

# Prerequisites
    # A server running Ubuntu 20.04.
    # A valid domain name pointed with your VPS.
    # A root password is setup on your server. 
    
# Getting Started
sudo apt-get update -y
sudo apt -y full-upgrade
sudo apt install build-essential checkinstall
sudo apt install ubuntu-restricted-extras
sudo apt install software-properties-common
sudo apt install software-properties-common
sudo apt upgrade -o APT::Get::Show-Upgraded=true
sudo apt-show-versions | grep upgradeable
sudo apt install apt-show-versions
sudo apt update -y
sudo apt-get upgrade -y 
sudo reboot

# Then, start the Apache and MariaDB service, and enable them to start at system reboot with the following command:
sudo systemctl start apache2
sudo systemctl start mariadb
sudo systemctl enable apache2
sudo systemctl enable mariadb

# If everything is fine, you should see the following output: 
Syntax OK

# Next, enable the GitScrum virtual host with the following comamnd: 
a2ensite gitscrum.conf

# Next, enable the Apache rewrite module and restart the Apache service to apply the changes:

a2enmod rewrite
systemctl restart apache2

# You can also verify the Apache service status using the following command:
systemctl status apache2

# Access the GitScrum Web Interface
# Go to http://gitscrum.localhost

