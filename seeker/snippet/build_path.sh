#date: 2024-04-22T17:02:38Z
#url: https://api.github.com/gists/04eebb9dd557c7f5bf101f1f688e2aa4
#owner: https://api.github.com/users/todaatsushi

#!/bin/bash

PATH=$2
SEARCH_TERM=$1

RESULT="$(rg -L "$SEARCH_TERM" "$2" | fzf)"
FILE="$(echo $RESULT | cut -d ":" -f 1 | xargs)"
SNIPPET="$(echo $RESULT | cut -d ":" -f 2 | xargs)"

python build_path.py "$SNIPPET" "$FILE"