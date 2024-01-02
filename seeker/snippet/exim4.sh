#date: 2024-01-02T17:01:53Z
#url: https://api.github.com/gists/315284717830b12c2000d1c8815f7a4e
#owner: https://api.github.com/users/mredisax

#!/bin/bash
exim_stats=$(eximstats /var/log/exim4/mainlog | head -11 | grep -E 'Received|Delivered|Rejects' | awk '{print $1"="$3}' | awk -v ORS=" " '{ print $0 }')
exim_queue=$(/usr/sbin/exim -bpc)

echo -n "$exim_stats" 
echo "Queue=$exim_queue"