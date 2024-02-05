#date: 2024-02-05T16:49:39Z
#url: https://api.github.com/gists/f84579eb007a9cbf5961b1821d994ea2
#owner: https://api.github.com/users/david-patchfox

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