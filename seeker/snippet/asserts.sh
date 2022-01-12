#date: 2022-01-12T17:00:17Z
#url: https://api.github.com/gists/26121c5a2d3e78428a0cb3d5af9b955e
#owner: https://api.github.com/users/garfieldnate

#!/bin/bash

pass() {
    message=$1;
    # bold green
    echo -e "$(tput setaf 2)$(tput bold)$message$(tput sgr 0)"
}

fail() {
    message=$1;
    # bold red
    echo -e "$(tput setaf 1)$(tput bold)$message$(tput sgr 0)"
}

# assert that output of cmd contains search_string
# example:
# >assert_outputs "docker run hello-world" 'Hello from Docker!' 'Docker installed successfully' 'Docker install failed'
assert_outputs() {
    cmd=$1;
    search_string=$2;
    success_msg=$3;
    fail_msg=$4;
    if $cmd | grep -q "$search_string";
    then
        pass "$success_msg";
    else
        fail "$fail_msg";
        exit 1;
    fi
}
