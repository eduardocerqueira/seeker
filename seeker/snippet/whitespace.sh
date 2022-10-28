#date: 2022-10-28T17:16:33Z
#url: https://api.github.com/gists/92b22a9c5b67c47c9e9e5e14da8dba30
#owner: https://api.github.com/users/CryptoFewka

#!/usr/bin/env bash
#
#set -x
#
# forked from jessebutryn/morse.sh
#######################################
declare -A whitespace
whitespace[0]='	 	 	 	 	'
whitespace[1]=' 	 	 	 	'
whitespace[2]='  	 	 	'
whitespace[3]='   	 	'
whitespace[4]='    	'
whitespace[5]='     '
whitespace[6]='	     '
whitespace[7]='	 	    '
whitespace[8]='	 	 	   '
whitespace[9]='	 	 	 	  '
whitespace[A]=' 	'
whitespace[B]='	    '
whitespace[C]='	  	  '
whitespace[D]='	   '
whitespace[E]=' '
whitespace[F]='  	  '
whitespace[G]='	 	  '
whitespace[H]='    '
whitespace[I]='  '
whitespace[J]=' 	 	 	'
whitespace[K]='	  	'
whitespace[L]=' 	   '
whitespace[M]='	 	'
whitespace[N]='	  '
whitespace[O]='	 	 	'
whitespace[P]=' 	 	  '
whitespace[Q]='	 	  	'
whitespace[R]=' 	  '
whitespace[S]='   '
whitespace[T]='	'
whitespace[U]='  	'
whitespace[V]='   	'
whitespace[W]=' 	 	'
whitespace[X]='	   	'
whitespace[Y]='	  	 	'
whitespace[Z]='	 	   '
whitespace[.]=' 	  	  	'
whitespace[,]='	 	   	 	'
whitespace[;]='	  	  	  '
whitespace[:]='	 	 	    '
whitespace[-]='	     	'
whitespace[/]='	   	  '
whitespace[\']=' 	 	 	 	  '
whitespace[\"]=' 	   	  '

#######################################
while read -rN1 c; do
	c=${c^}
	if [[ $c == $'\n' ]]; then
		printf '\n'
	elif [[ $c == ' ' ]]; then
		printf '       '
	else
		printf '%s   ' "${whitespace[$c]}"
	fi
done
