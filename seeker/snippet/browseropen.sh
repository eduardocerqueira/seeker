#date: 2023-01-31T17:02:58Z
#url: https://api.github.com/gists/09f5937f7cc0f1041d6b35797c00e230
#owner: https://api.github.com/users/arik181

#!/usr/bin/env sh

browser=w3m
feedreader=snownews
dt=`date +%F`

touch $dt
echo $1 >> ${HOME}/Git/history/${dt}
zellij --session $feedreader r -c -- $browser $1
cd ${HOME}/Git/history
git add *
git commit -m "Adding links for ${dt}"
git push origin main