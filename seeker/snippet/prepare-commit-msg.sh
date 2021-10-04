#date: 2021-10-04T17:04:46Z
#url: https://api.github.com/gists/1af7f8c31db89733fa53c33ff5c1b4b6
#owner: https://api.github.com/users/jon-ruckwood

#!/bin/sh

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2
SHA1=$3

git interpret-trailers --in-place "$COMMIT_MSG_FILE"