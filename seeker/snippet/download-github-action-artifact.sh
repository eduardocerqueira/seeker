#date: 2024-01-25T16:48:10Z
#url: https://api.github.com/gists/ad0bdfbba4db1be3c1f613031f8860cb
#owner: https://api.github.com/users/bucherfa

#!/usr/bin/env bash

# load environment variables from .env file
set -a
source .env
set +a

# get the artifact URL
ARTIFACT_URL=$(curl --silent --location \
  --header "Accept: application/vnd.github+json" \
  --header "Authorization: "**********"
  --header "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/artifacts \
  | jq '[.artifacts[] | {name : .name, archive_download_url : .archive_download_url}]' \
  | jq --arg artifact_name $ARTIFACT_NAME --raw-output '.[] | select (.name == $artifact_name) | .archive_download_url' \
  | head --lines=1)

# download the artifact
curl --location \
  --header "Accept: application/vnd.github+json" \
  --header "Authorization: "**********"
  --header "X-GitHub-Api-Version: 2022-11-28" \
  --output $ARTIFACT_NAME.zip \
  $ARTIFACT_URL

# unzip to directory
unzip $ARTIFACT_NAME.zip -d $DIRECTORY_DESTINATION
_NAME.zip -d $DIRECTORY_DESTINATION
