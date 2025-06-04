#date: 2025-06-04T16:53:15Z
#url: https://api.github.com/gists/b66e7693243698250057a61ec87b06da
#owner: https://api.github.com/users/arianvp

#!/bin/bash

# Calculates the median amount of seconds since a file has been changed
# by taking the time between creation and last edited for eadch line
# 0 means the file was never changed since creation

for file in $(git ls-files '*.c' '*.h'); do
  if [ -f "$file" ]; then
    times=($(git blame --line-porcelain "$file" | grep '^author-time ' | awk '{print $2}' | sort -n))
    count=${#times[@]}
    if [ "$count" -gt 0 ]; then
      t_create=${times[0]}
      mid=$((count / 2))
      if (( count % 2 == 0 )); then
        t_median=$(( (times[mid-1] + times[mid]) / 2 ))
      else
        t_median=${times[mid]}
      fi
      touchedness=$(( t_median - t_create ))
      echo "$touchedness $t_create $t_median $file"
    fi
  fi
done | sort -nr