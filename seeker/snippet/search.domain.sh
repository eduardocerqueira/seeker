#date: 2023-06-12T17:00:35Z
#url: https://api.github.com/gists/c8215633eb18b00e37b0d02420a2ecd4
#owner: https://api.github.com/users/OCharnyshevich

#!/bin/bash 
  
if [ "$#" == "0" ]; then 
    echo "You need tu supply at least one argument!" 
    exit 1
fi 
 
DOMAINS=( '.com')
 
ELEMENTS=${#DOMAINS[@]} 
 
while (( "$#" )); do 
 
  for (( i=0;i<$ELEMENTS;i++)); do 
      whois $1${DOMAINS[${i}]} | grep -qciE '^No match for domain|^Domain not found|Domain is not registered|^NOT FOUND$|^No Data Found$|is available for registration$|^The queried object does not exist: DOMAIN NOT FOUND$'
	  if [ $? -eq 0 ]; then 
	      echo "$1${DOMAINS[${i}]} : AVAILABLE!" 
	  else 
	      echo "$1${DOMAINS[${i}]} : not available" 
	  fi 
  done 
 
shift 
 
done