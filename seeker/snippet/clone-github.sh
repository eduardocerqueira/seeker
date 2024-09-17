#date: 2024-09-17T16:55:11Z
#url: https://api.github.com/gists/ecec01f35abce4af34a2409af65dc85a
#owner: https://api.github.com/users/dleslie

#!/bin/bash

username="${username:=dleslie}"

function git_clone_or_update {
    local url=$1
    local name="${url##*/}";
    local name="${name%.git}";
    if [ -d $name ]; then
        pushd $name
        git pull
        popd
    else
        git clone --recursive $url $name
    fi
}

found_urls=""
function get_type {
    local type=$1
    local page=$2
    found_urls=`curl -s https://api.github.com/users/$username/$type?per_page=100\&page=$page | jq -c '.[] | .ssh_url'`
}

function fetch_type {
    local type=$1
    local root=$2
    local index=1

    mkdir -p $root
    pushd $root
    get_type $type $index
    while [ ! -z "$found_urls" ]; do
        for url in $found_urls; do
            url=`echo $url | tr -d \"`;
            git_clone_or_update $url;
        done
        index=$((index+1));
        get_type $type $index
    done
    popd
}

fetch_type repos $username
fetch_type starred starred