#date: 2022-04-19T17:11:08Z
#url: https://api.github.com/gists/5ac9be208e021448aa032c6b4397a5b4
#owner: https://api.github.com/users/kaancceylan

#!/bin/bash

spaces() {
  for ((i=0; i<$1; i++)) ; do
    echo -n " "
  done 
}

stars() {
  for ((i=0; i<$1; i++)) ; do
    echo -n "*"
  done
  echo ""
}

display_tree() {
    local rows=$1
    local columns=$2

    for (( r=1; r <= $rows; r++ )); do
      s=$(( (((columns-1) * (r-1)/(rows-1) + 1)/2)*2 +1 ))
      spaces $(((columns-s)/2))
      stars $s
    done
}

if [ $# -eq 2 ]; then
    display_tree $1 $2
else
    echo "You need to enter row and column numbers"
fi