#date: 2022-09-16T22:05:03Z
#url: https://api.github.com/gists/8303ee3d8e8d7bcd225212fb19913947
#owner: https://api.github.com/users/malkab

#!/bin/bash

IN="key1#key2"

IFS='#' read -ra ADDR <<< "$IN"

for i in "${ADDR[@]}"; do

  echo $i

done

unset IFS
