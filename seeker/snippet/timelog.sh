#date: 2023-02-01T16:36:26Z
#url: https://api.github.com/gists/8b8fa9c93160cb15a7ed7de3c8f435aa
#owner: https://api.github.com/users/ssj71

#!/bin/bash
#spencer
# just run this when i3lock starts

d=$(date)
sleep 30
if killall -0 i3lock;
then
    echo $d screen locked >> ~/timecard
else
    #it didn't stay locked long enough, don't count it
    exit
fi
while :
do
    sleep 30
    if ! killall -0 i3lock;
    then
        echo $(date) screen unlocked >> ~/timecard
        exit
    fi
done