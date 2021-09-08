#date: 2021-09-08T17:09:21Z
#url: https://api.github.com/gists/076a0a6e1184a000bc1b77ac299c049b
#owner: https://api.github.com/users/Iskander0

#!/bin/bash 
# GETS APPROX. VIEWERS CONNECTED TO SERVER
while :
do
	netstat -tn 2>/dev/null | grep :80 | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | head  > ips.txt
	wc -l /var/www/ips.txt | cut -d ' ' -f 1 > viewers.txt
	sleep 5
done
