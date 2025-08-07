#date: 2025-08-07T17:09:16Z
#url: https://api.github.com/gists/9f69cb41bfb665802fb26672acd06927
#owner: https://api.github.com/users/ZacheryFaria

#!/bin/bash

git checkout master
git pull
git fetch -p && git branch -vv | awk '/: gone]/{print $1}' | xargs git branch -D