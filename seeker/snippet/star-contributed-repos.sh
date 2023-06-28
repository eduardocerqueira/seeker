#date: 2023-06-28T16:43:17Z
#url: https://api.github.com/gists/098db6fb8da22864a8bfbb6e2f1c1d4f
#owner: https://api.github.com/users/Vikram710

#!/bin/bash
# Usage: "**********"
echo $1
echo $2

CONTRIBUTORS_URLS=$(curl -s  -H "Authorization: "**********"://api.github.com/orgs/$2/repos?per_page=200" | jq -r '.[].contributors_url')
for C_URL in $CONTRIBUTORS_URLS
    do 
        CONTRIBUTORS=$(curl -s  -H "Authorization: "**********"
        if echo $CONTRIBUTORS | grep -iqF $3; then
            REPO=$(echo $C_URL | cut -d'/' -f6)
            echo "STARRING $REPO"
            curl -L \
            -s \
            -X PUT \
            -H "Accept: application/vnd.github+json" \
            -H "Authorization: Bearer $1" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            https://api.github.com/user/starred/$2/$REPO
        fi
    done
         https://api.github.com/user/starred/$2/$REPO
        fi
    done
