#date: 2024-03-07T16:54:03Z
#url: https://api.github.com/gists/0655945874a6df3450806ef93ead9886
#owner: https://api.github.com/users/GammaGames

#!/usr/bin/env bash

_val=`curl -Isk $2 | grep HTTP | cut -d ' ' -f2`
echo "$(date)    $_val"
_diff=0

while :
do
  _val2=`curl -Isk $2 | grep HTTP | cut -d ' ' -f2`
  if [ "$_val" != "$_val2" ]; then
    echo "$(date)    $_val2"
    _diff=1
  else if [ $_diff -eq 1 ]; then
      echo "$(date)    $_val2"
      _diff=0
    fi
  fi
  sleep $1
done
