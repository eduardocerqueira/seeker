#date: 2022-09-16T21:12:50Z
#url: https://api.github.com/gists/a38c7190c20561615768aa6b9bce6dec
#owner: https://api.github.com/users/sidneyflima

#!/bin/bash

# extract linux ubuntu wsl ip
export WSL_IP=$(echo $(ifconfig eth0 | awk -e '/inet \w*/ {print $2}'))
export WSL_IP_ENTRY=$(echo $WSL_IP wsl-server)

# write content into this files
[[ ! -z $WSL_IP ]] && echo $WSL_IP > ~/.wsl.ip
[[ ! -z $WSL_IP_ENTRY ]] && echo $WSL_IP_ENTRY > ~/.wsl.entry
