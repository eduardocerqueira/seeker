#date: 2022-11-28T17:09:23Z
#url: https://api.github.com/gists/74be9dd0a76332b1b49c7d976f7421c6
#owner: https://api.github.com/users/r3w0p

#!/bin/bash

# autogit v1.0

# ---

REPO_LOCAL="/path/to/repo"
REPO_REMOTE="<organisation>/<repo_name>.git"
REPO_BRANCH="<branch>"

USERNAME="<username>"
PASSWORD= "**********"

PATH_LOG_FILE="/path/to/autogit.log"
DATE_COMMIT="$(date +"%Y-%m-%d %H:%M")"

# ---

mkdir -p "${PATH_LOG_FILE%/*}/" && rm -f $PATH_LOG_FILE && touch $PATH_LOG_FILE
cd $REPO_LOCAL

printf "$DATE_COMMIT" >> $PATH_LOG_FILE
printf "\n" >> $PATH_LOG_FILE

git add * >> $PATH_LOG_FILE
printf "\n" >> $PATH_LOG_FILE

git commit -a -m "autogit ($DATE_COMMIT)" >> $PATH_LOG_FILE
printf "\n" >> $PATH_LOG_FILE

git push -u https: "**********":$PASSWORD@github.com/$REPO_REMOTE $REPO_BRANCH >> $PATH_LOG_FILE
printf "\n" >> $PATH_LOG_FILE
