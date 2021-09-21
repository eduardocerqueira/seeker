#date: 2021-09-21T17:10:17Z
#url: https://api.github.com/gists/e6e0384c97b9b29db5be5aab746ee706
#owner: https://api.github.com/users/Pzoom522

#! /bin/bash

lang_list=(en de es fr hi pl)

ban=0
for i in "${!lang_list[@]}"
do
  ban=$((ban + 1))
  echo "$ban"
	for j in "${!lang_list[@]}"	
	do
		if ((j >= ban)); then
	    echo "${lang_list[$i]}"-"${lang_list[$j]}"
	  fi
	done
done
