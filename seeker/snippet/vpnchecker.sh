#date: 2021-10-14T17:04:41Z
#url: https://api.github.com/gists/702edd1ddf3185e385f40c3213553ef9
#owner: https://api.github.com/users/root-tanishq

#!/bin/bash
Extension download link - https://extensions.gnome.org/extension/1176/argos/
Paste this file in ~/.config/argos/vpnchecker.sh
if [ $(ifconfig tun0 2>/dev/null 1>/dev/null ; echo $?) == 0 ];then 
	echo "⚡`ifconfig tun0 | grep netmask | cut -f 10 -d ' '`"; 
else 
	hostname -I | awk '{print $1}'; 
fi