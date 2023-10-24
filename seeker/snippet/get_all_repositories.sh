#date: 2023-10-24T16:51:32Z
#url: https://api.github.com/gists/6d9d16cae9723806536504b5f4ba0864
#owner: https://api.github.com/users/btmash

#!/bin/bash

REPOSITORIES=$(aws codecommit list-repositories --sort-by repositoryName --query 'repositories[].repositoryName' --output text)

if ! command -v parallel &> /dev/null
then
  for REPO in $REPOSITORIES
  do
    bash ./get_repo.sh $REPO
  done
else
  parallel -j 4 bash ./get_repo.sh {} ::: $REPOSITORIES

fi
