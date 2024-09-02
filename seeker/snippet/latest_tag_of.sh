#date: 2024-09-02T16:57:16Z
#url: https://api.github.com/gists/101f6c16d291d8d8048ab9dd532cd7fc
#owner: https://api.github.com/users/alpha-tango-kilo

#!/bin/sh

# This script expects one argument, being the "user/repo" slug for a GitHub repository
# e.g. alpha-tango-kilo/imdb-id

GITHUB_PAT="<Your PAT here>"
REPO=$1

curl -H "Authorization: "**********"://api.github.com/repos/$REPO/releases/latest" \
    | jq --exit-status --raw-output '.tag_name'tput '.tag_name'