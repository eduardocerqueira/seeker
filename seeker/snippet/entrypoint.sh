#date: 2023-04-06T17:08:28Z
#url: https://api.github.com/gists/7e0f71dd7fc33fc5193c1b3de47ab584
#owner: https://api.github.com/users/BlackthornYugen

#!/usr/bin/env bash
random_password() {
    echo -n "jumpuser password: "**********"
    dd if=/dev/random count=$(($1 * 2)) bs=1 2> /dev/null | base64 | tr -d '/=+' | head -c "$1" | tee /dev/stderr
    echo > /dev/stderr
}

chpasswd <<< "jumpuser: "**********"
while sleep 3 ; do 
  /usr/sbin/sshd -D -p 22
doneD -p 22
done