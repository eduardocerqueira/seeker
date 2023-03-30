#date: 2023-03-30T16:55:53Z
#url: https://api.github.com/gists/309c6f5673b98f4af80946952698dfdc
#owner: https://api.github.com/users/vagnerd

#!/bin/bash

while [ 0 ]; do
	wget -q https://gist.githubusercontent.com/vagnerd/XXX/raw/YYY/ips -O /tmp/ips
	cat /tmp/ips | awk '{ print "route add " $1 " gw X.X.X.X"}' > /tmp/ips.sh
	sh /tmp/ips.sh
	sleep 3600
done

### rc-local
## https://www.cyberciti.biz/faq/how-to-enable-rc-local-shell-script-on-systemd-while-booting-linux-system/
#
#nohup /root/update-route-with-gist.sh &
#exit 0