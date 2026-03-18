#date: 2026-03-18T17:50:19Z
#url: https://api.github.com/gists/96aa39cc384b17a907d9ca7b9e55af6e
#owner: https://api.github.com/users/WumboSpasm

#!/bin/bash
cd "$1"
for f in \$I*; do
	raw_content=$(cat $f | tr -d '\0')
	hashed_name=$(echo $f | sed s/./R/2)
	real_name=${raw_content##*\\}
	echo "$hashed_name -> $real_name"
done