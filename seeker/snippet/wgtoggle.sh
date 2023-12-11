#date: 2023-12-11T17:09:17Z
#url: https://api.github.com/gists/a6533c573fb19ecaa808482bd668b5fa
#owner: https://api.github.com/users/wernerjoss

#!/bin/bash
#   toggle wireguard interface status, if connected: disconnect, and vice versa 
#   wernerjoss 11.11.23

wgclient="orion"    # default

while getopts ":c:v:" opt; do
	case $opt in 
	c)
		wgclient="$OPTARG"
		;;
	v)
		verbose="$OPTARG"
		;;
	\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1
		;;
	esac
done

if [ ! -z $verbose ]; then
    echo $wgclient
fi

status=$(nmcli -o d | grep $wgclient | awk '{print $1}')

if [ ! -z $verbose ]; then
    echo $status
fi

if [ $status == $wgclient ]; then
    echo "$wgclient is up, disconnecting"
    wg-quick down $wgclient
    
else
    echo "$wgclient is down, connecting"
    wg-quick up $wgclient
fi
