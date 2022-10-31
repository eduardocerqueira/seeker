#date: 2022-10-31T17:24:54Z
#url: https://api.github.com/gists/770e09e650f9ec2422f77ff64c623268
#owner: https://api.github.com/users/AndreFCAmorim

#!/bin/bash
apt update
apt upgrade -y
wget https://prdownloads.sourceforge.net/webadmin/webmin_1.981_all.deb
apt install ./webmin_1.981_all.deb
echo What is the port number:
read portnumber
sed -i 's/port=10000/port='$portnumber'/g' /etc/webmin/miniserv.conf
/etc/init.d/webmin restart
echo Installation done!
ip=$(ip addr|grep 'inet '|grep global|head -n1|awk '{print $2}'|cut -f1 -d/)
echo In your browser: https://$ip:$portnumber