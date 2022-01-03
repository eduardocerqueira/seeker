#date: 2022-01-03T17:21:21Z
#url: https://api.github.com/gists/72d8136069ca1f1667127bf239e9c358
#owner: https://api.github.com/users/a11ce

#!/bin/bash

# no set -e because gh might 404
set -uo pipefail
IFS=$'\n\t'

if test "$#" -ne 2; then
    echo "run as ./github-add-collabs.sh user/repo usernames.txt"
    exit
fi

while read username; do
    echo "adding $username..."
    gh api --silent -XPUT "repos/$1/collaborators/$username" -f permission=push
    if test "$?" -ne 0; then
        echo -e "\033[0;31m!!!! could not find $username\033[0m"
    fi
done < $2
