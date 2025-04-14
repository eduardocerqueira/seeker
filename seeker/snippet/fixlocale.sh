#date: 2025-04-14T16:35:56Z
#url: https://api.github.com/gists/d8db86b19246c5b46b5c263c21da67e7
#owner: https://api.github.com/users/vtf6259

#!/bin/bash

echo "LC_ALL=en_US.UTF-8" | sudo tee -a /etc/environment
echo "en_US.UTF-8 UTF-8" | sudo tee -a /etc/locale.gen
echo "LANG=en_US.UTF-8" | sudo tee -a /etc/locale.conf
sudo locale-gen en_US.UTF-8