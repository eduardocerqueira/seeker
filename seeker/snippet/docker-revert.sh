#date: 2023-03-09T16:54:41Z
#url: https://api.github.com/gists/b612ba89c3ce75b0b3a8395aa5a75ff3
#owner: https://api.github.com/users/prafiles

#!/bin/sh
sudo apt-get install docker-ce=5:20.10.23~3-0~ubuntu-jammy docker-ce-cli=5:20.10.23~3-0~ubuntu-jammy containerd.io=1.6.18-1 docker-compose-plugin=2.16.0-1~ubuntu.22.04~jammy
sudo apt-mark hold docker-ce docker-ce-cli containerd.io docker-compose-plugin