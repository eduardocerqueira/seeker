#date: 2022-10-19T17:27:06Z
#url: https://api.github.com/gists/bbd9e34a49d4434d31d9c879adb08b67
#owner: https://api.github.com/users/react-project-base

#!/bin/bash
ip link add dummy1 type dummy
ip link set dummy1 up
ip -6 addr add [the IPv6 block we want to announce] dev dummy1