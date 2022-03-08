#date: 2022-03-08T17:07:11Z
#url: https://api.github.com/gists/e877e87e8a270e84d8598e4b34b5ae60
#owner: https://api.github.com/users/taboubim

#!/bin/sh

readarray array <<< $( cat "$@" )

mkdir -p ~/git && cd ~/git

for element in ${array[@]}
do
  echo "clonning $element"
  git clone $element
done
