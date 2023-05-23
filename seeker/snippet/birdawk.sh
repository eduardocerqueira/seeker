#date: 2023-05-23T16:43:51Z
#url: https://api.github.com/gists/cb922d06f8a59b6624a278bfa4c64f53
#owner: https://api.github.com/users/drakonstein

#!/bin/bash

inround=false
exec 5<>/dev/tcp/35.231.207.196/15830
while true; do
  if $inround; then
    awk 'BEGIN {coords=""} /B|^$/ {
      if(NF == 0)
        exit
      for(i=2;i<=NF;i++) {
        if($i == "B")
          coords = coords";("i-2","$1")"
      }
    } END {print substr(coords,2)}' <&5 >&5 && inround=false || echo huh
  else
    read row line <&5 || break
    (( round == 200 )) && echo "$row $line" | grep flag && exit 0
    [[ "$row" == "Round:" ]] && inround=true && round=$line
  fi
done
echo "Got to round: $round"
exit 1