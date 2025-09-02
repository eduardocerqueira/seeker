#date: 2025-09-02T16:45:54Z
#url: https://api.github.com/gists/77032cedfea289c88f0ee83c88f2c0d7
#owner: https://api.github.com/users/oriionn

#!/bin/bash

if [ -z "$1" ]; then
    echo "No file specified"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "File doesn't exists"
    exit 1
fi

data=$(curl -s -F "files[]=@$1" https://uguu.se/upload)
success=$(echo $data | jq ".success")

if [ ! $success = 'true' ]; then
    echo "An error occurred when sending the file"
    exit 1
fi

url=$(echo $data | jq -r ".files | .[] | .url")
echo $url
