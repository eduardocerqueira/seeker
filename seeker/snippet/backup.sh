#date: 2024-10-02T16:57:38Z
#url: https://api.github.com/gists/52e59cbeb6951659ecbeb41d1bf29560
#owner: https://api.github.com/users/ciscolyon69

#!/bin/sh

# To run before
# git config core.sshCommand "ssh -i /config/.ssh/id_rsa -o 'StrictHostKeyChecking=no' -F /dev/null"

HA_VERSION=`cat .HA_VERSION`
COMMIT_CURRENT_DATE=$(date +'%d-%m-%Y %H:%M:%S')
COMMIT_MESSAGE="[$HA_VERSION]: $COMMIT_CURRENT_DATE"

echo "$COMMIT_MESSAGE"

git add .
git commit -m "$COMMIT_MESSAGE"
git push