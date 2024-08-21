#date: 2024-08-21T17:07:46Z
#url: https://api.github.com/gists/6b0fedee86ff450df5693c253d4de3e1
#owner: https://api.github.com/users/yetanotherchris

# Tested on Ubuntu, use 'apt-get install jq'

cat ./bookmarks.json | jq '.. | select(.type?=="text/x-moz-place") | "\(.title?), \(.uri)"?'
cat ./bookmarks.json | jq -r '.. | select(.type?=="text/x-moz-place") | "- [\(.title?)](\(.uri?))"'  > uris.md

# The second prints a markdown file using the title and uri