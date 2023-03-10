#date: 2023-03-10T16:59:32Z
#url: https://api.github.com/gists/0ebe6167608577c6547336de371d5b78
#owner: https://api.github.com/users/rabestro

#!/usr/bin/env bash

main () {
    local -r phrase=${1@L}
    local symbols=${phrase//[[:space:]-]/}

    for letter in {a..z}
    do
       symbols="${symbols/$letter/}"
    done

    [[ -z $symbols ]] && echo 'true' || echo 'false'
}

main "$@"
