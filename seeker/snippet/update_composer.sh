#date: 2025-02-24T17:00:56Z
#url: https://api.github.com/gists/430d5a756952dced693269d70fd33bd3
#owner: https://api.github.com/users/lumenpink

composer require $(composer show -s --format=json | jq '.requires | keys | map(.+" ") | add' -r)