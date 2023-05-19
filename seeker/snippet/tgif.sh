#date: 2023-05-19T17:02:17Z
#url: https://api.github.com/gists/56e857ffde294d3d5a8ad065e62bff40
#owner: https://api.github.com/users/chancehl

#!/bin/bash
FILE="/usr/share/dict/words"

# get random element fn
function ref {
  declare -a array=("$@")
  r=$((RANDOM % ${#array[@]}))
  printf "%s\n" "${array[$r]}"
}

# pick random words
T=$(ref $(grep "^[t]" $FILE | grep -E '^.{3,6}$'))
G=$(ref $(grep "^[g]" $FILE | grep -E '^.{3,6}$'))
I=$(ref $(grep "^[i]" $FILE | grep -E '^.{3,6}$'))
F=$(ref $(grep "^[f]" $FILE | grep -E '^.{3,6}$'))

# print
echo "$T $G $I $F"
