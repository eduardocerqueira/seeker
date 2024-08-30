#date: 2024-08-30T17:10:15Z
#url: https://api.github.com/gists/f1825a7e2c54c2c6e46095f95c697285
#owner: https://api.github.com/users/genzj

#!/bin/bash

# ref:
#   https://askubuntu.com/a/1167767
#   https://manpages.ubuntu.com/manpages/bionic/man5/NetworkManager.conf.5.html#connectivity%20section

sudo cp --backup=t /etc/NetworkManager/NetworkManager.conf /etc/NetworkManager/NetworkManager.conf.backup
echo -e "\n[connectivity]\nuri=\n" | sudo tee -a /etc/NetworkManager/NetworkManager.conf
sudo systemctl restart NetworkManager.service