#date: 2022-09-28T17:17:15Z
#url: https://api.github.com/gists/8b2dc08f091f14122c9b47f53968ad94
#owner: https://api.github.com/users/ps-jessejjohnson

#!/usr/local/bin/bash

pgrep -f rosetta | xargs kill
pgrep -f node | xargs kill -9
pgrep -f rake | xargs kill -9
sleep 10
for proc in node rake rosetta ;do
    echo Checking for $proc processes
    pgrep -f $proc
done