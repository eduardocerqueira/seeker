#date: 2021-10-13T17:14:04Z
#url: https://api.github.com/gists/d4624aab48f3541cf125622a1e84ca21
#owner: https://api.github.com/users/afahitech

#!/bin/sh

sudo apt update
sudo apt install openjdk-11-jdk
java -version
wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
sudo apt update
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 9B7D32F2D50582E6
systemctl status jenkins
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 9B7D32F2D50582E6
sudo apt install jenkins
systemctl status jenkins
sudo ufw allow proto tcp from 192.168.121.0/24 to any port 8080
sudo ufw allow 8080
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
