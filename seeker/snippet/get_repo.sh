#date: 2023-10-24T16:51:32Z
#url: https://api.github.com/gists/6d9d16cae9723806536504b5f4ba0864
#owner: https://api.github.com/users/btmash

#!/bin/bash

if [ -z "$1" ];
then
  echo 'No repository name supplied'
  exit 1
fi

REPO=$1
echo $REPO
DETAILS=$(aws codecommit get-repository --repository-name $REPO)
REPONAME=$(echo $DETAILS | jq '.repositoryMetadata.repositoryName' | tr -d '"')
REPO_SSH_PATH=$(echo $DETAILS | jq '.repositoryMetadata.cloneUrlSsh' | tr -d '"')
rm -rf $REPONAME
git clone $REPO_SSH_PATH $REPONAME
echo $REPONAME
echo $REPO_SSH_PATH
