#date: 2022-11-24T17:11:04Z
#url: https://api.github.com/gists/78dab92e661be26bd12ef0f5b3e1308f
#owner: https://api.github.com/users/cloudybdone

If you need any help related to GitLab Installation Service on Any Linux contact with me:

Telegram: https://t.me/Cloudybdone
WhatsApp: https://wa.link/3j794g
Skype: https://join.skype.com/invite/vLFaKHx...
Email: cloudybdone@gmail.com
Linkedin: https://www.linkedin.com/in/cloudybdone/
Facebook: https://www.facebook.com/cloudybdone/
About Me: https://about.me/cloudybdone

YouTube Playlist: 

#!/bin/sh

# Installing the Dependencies
sudo apt update
sudo apt install ca-certificates curl openssh-server postfix tzdata perl

# Installing GitLab
cd /tmp
curl -LO https://packages.gitlab.com/install/repositories/gitlab/gitlab-ce/script.deb.sh
less /tmp/script.deb.sh
sudo bash /tmp/script.deb.sh
sudo apt install gitlab-ce

# Adjusting the Firewall Rules
sudo ufw status
sudo ufw allow http
sudo ufw allow https
sudo ufw allow OpenSSH

# Editing the GitLab Configuration File
sudo nano /etc/gitlab
sudo gitlab-ctl reconfigure

# Performing Initial Configuration Through the Web Interface
https://your_domain

 "**********"# "**********"  "**********"G "**********"i "**********"t "**********"L "**********"a "**********"b "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"s "**********"  "**********"a "**********"n "**********"  "**********"i "**********"n "**********"i "**********"t "**********"i "**********"a "**********"l "**********"  "**********"s "**********"e "**********"c "**********"u "**********"r "**********"e "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"f "**********"o "**********"r "**********"  "**********"y "**********"o "**********"u "**********". "**********"  "**********"I "**********"t "**********"  "**********"i "**********"s "**********"  "**********"s "**********"t "**********"o "**********"r "**********"e "**********"d "**********"  "**********"i "**********"n "**********"  "**********"a "**********"  "**********"f "**********"o "**********"l "**********"d "**********"e "**********"r "**********"  "**********"t "**********"h "**********"a "**********"t "**********"  "**********"y "**********"o "**********"u "**********"  "**********"c "**********"a "**********"n "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"  "**********"a "**********"s "**********"  "**********"a "**********"n "**********"  "**********"a "**********"d "**********"m "**********"i "**********"n "**********"i "**********"s "**********"t "**********"r "**********"a "**********"t "**********"i "**********"v "**********"e "**********"  "**********"s "**********"u "**********"d "**********"o "**********"  "**********"u "**********"s "**********"e "**********"r "**********": "**********"
