#date: 2021-11-01T16:50:46Z
#url: https://api.github.com/gists/0e740b2d45f76a68498f1c6682a09ce5
#owner: https://api.github.com/users/AnalistaDesarrolloABAP2

#!/bin/sh

readarray array <<< $( cat "$@" )

mkdir -p ~/git && cd ~/git

for element in ${array[@]}
do
  echo "clonning $element"
  git clone $element
done
