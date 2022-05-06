#date: 2022-05-06T17:13:53Z
#url: https://api.github.com/gists/bfd364c1bb19fcc1b9769e5aba6f104e
#owner: https://api.github.com/users/dannysauer

#!/bin/bash

ORG=Kong
declare -A externals

printf "repo,user,external?\n"
for REPO in $(gh api --paginate "/orgs/$org/repos?type=public&sort=full_name&per_page=100" \
              | jq -r '.[] | select( .fork == false ) | .name')
do
  echo "processing $REPO" >&2
  for USER in $(gh api --paginate "/repos/$org/$REPO/contributors?per_page=100" | jq -r '.[] | .login')
  do
    if [[ ${externals[$USER]+_} ]]
    then
      # alredy-chcked user
      :
    else
      if gh api "/orgs/$org/members/$USER" > /dev/null 2>&1
      then
        externals[$USER]=0
      else
        externals[$USER]=1
      fi
    fi
    printf "$REPO,$USER,${externals[$USER]}\n"
  done
done

count=0
total=0
for c in "${!externals[@]}"
do
  (( total++ ))
  (( count+= ${externals[$c]} ))
done
printf "$count external contributors out of $total total\n" >&2
