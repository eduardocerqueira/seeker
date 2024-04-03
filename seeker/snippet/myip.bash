#date: 2024-04-03T16:41:40Z
#url: https://api.github.com/gists/ec18b2198c49e2b0422717de5ab1c9c8
#owner: https://api.github.com/users/bmatthewshea

#!/bin/bash
printf "Gateway:\n"; route | grep "^default" | cut -d " " -f 10
# Adjust network interface number (here I use interface #2 "2:" for lan address)
printf "Private:\n"; ip -o ad | grep "2:" | grep "inet " | cut -d " " -f 7
printf "Public:\n"; dig +short myip.opendns.com @resolver1.opendns.com
