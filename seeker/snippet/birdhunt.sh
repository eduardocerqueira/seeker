#date: 2023-05-23T16:43:51Z
#url: https://api.github.com/gists/cb922d06f8a59b6624a278bfa4c64f53
#owner: https://api.github.com/users/drakonstein

#!/bin/bash

start=false
coords=
exec 5<>/dev/tcp/35.231.207.196/15830
while read row line; do
# Loop through quickly to find the start of the first round
  if ! $start; then
    [[ "$row" == "Round:" ]] && start=true && round=$line
    continue
  fi

# Find the right time to submit the coordinates
  [[ -z "$row" && ! -z "$coords" ]] && echo ${coords:1} >&5 && coords= && continue

# Check for which column the B is in
  [[ "$line" == *"B"* ]] && column=0 && for line2 in $line; do
    [[ $line2 == "B" ]] && coords="$coords;($column,$row)"
    column=$(( column + 1))
  done <<< "$line" && continue
  [[ "$row" == "Round:" ]] && round=$line && continue
  (( round == 200 )) && echo "$row $line" | grep flag && exit 0
done <&5
echo "Got to round: $round"
exit 1