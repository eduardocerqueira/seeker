#date: 2022-07-20T17:17:06Z
#url: https://api.github.com/gists/1dc740084ac1a36bdadfee95b534e230
#owner: https://api.github.com/users/aboron

#/bin/bash
while [ "$key" != "q" ]
do
        clear
        uptime
        echo "`$* | head -n 35`"
        read -n 1 -t 2 -s -r key
done
