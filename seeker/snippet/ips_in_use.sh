#date: 2022-07-20T17:15:13Z
#url: https://api.github.com/gists/95a0d41241074a9bbdd864ca5e9901f9
#owner: https://api.github.com/users/aboron

#!/usr/bin/bash

if [ "$#" -ne 1 ]; then
    vmadm list -p -o uuid | while read line ;do
    vmadm get $line | /usr/xpg4/bin/grep -E 'mac|"ip"|zonename|interface|alias'
    echo
    done
    exit 0
fi

vmadm list -p -o uuid | while read line ;do
results=`vmadm get $line | /usr/xpg4/bin/grep -E '"ip"' | /usr/xpg4/bin/grep -c $1`
if [ $results -ge 1 ]; then
    vmadm get $line | /usr/xpg4/bin/grep -E 'zonename|alias'
fi
done
exit 0
