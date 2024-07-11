#date: 2024-07-11T16:47:55Z
#url: https://api.github.com/gists/6311b18793679e976a44f92ef9fa7c41
#owner: https://api.github.com/users/themactep

#!/bin/sh
# IPC button catcher
# Paul Philippov <paul@themactep.com>
# 2024-07-01: Initial release

GPIO_MAX=95
EXCLUDE="10 16 17 18 49 54 55 56 57 58"

for i in $(seq 0 $GPIO_MAX); do
    echo $EXCLUDE | grep -e "\b$i\b" >/dev/null && continue
    echo gpio input $i
done

gpio list > /tmp/old
while :; do
    gpio list > /tmp/new
    diff /tmp/old /tmp/new
    sleep 1
done

exit 0