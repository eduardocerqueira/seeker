#date: 2022-10-28T17:16:58Z
#url: https://api.github.com/gists/5f6b18efc51b9bde8cd325fa7c38b973
#owner: https://api.github.com/users/devkabir

#!/bin/bash

if [ -z "$1" ]; then
    echo "waiting for the following arguments: username + max-page-number"
    exit 1
else
    name=$1
fi

if [ -z "$2" ]; then 
    max=2
else
    max=$2
fi

cntx="users"
page=1

echo $name
echo $max
echo $cntx
echo $page

until (( $page -lt $max ))
do 
    curl "https://api.github.com/$cntx/$name/repos?page=$page&per_page=100" | grep -e 'clone_url*' | cut -d \" -f 4 | xargs -L1 git clone
    $page=$page+1
done

exit 0