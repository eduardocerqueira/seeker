#date: 2024-08-27T17:06:09Z
#url: https://api.github.com/gists/d6a4751a624b92d826c13624930f3354
#owner: https://api.github.com/users/thiagomiranda3

#!/bin/bash

#Whenever you clone a repo, you do not clone all of its branches by default.
#If you wish to do so, use the following script:

for branch in `git branch -a | grep remotes | grep -v HEAD | grep -v master `; do
   git branch --track ${branch#remotes/origin/} $branch
done