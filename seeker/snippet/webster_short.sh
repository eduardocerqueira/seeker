#date: 2022-07-06T17:20:20Z
#url: https://api.github.com/gists/43205cac1e770c54c14ac618303be65b
#owner: https://api.github.com/users/R4wm

#!/bin/bash

key=$(cat ~/.webster/config.json | jq .key | sed s/\"//g)
curl -s "https://www.dictionaryapi.com/api/v3/references/collegiate/json/$1?key=$key" | jq .[0].shortdef