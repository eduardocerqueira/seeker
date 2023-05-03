#date: 2023-05-03T16:55:13Z
#url: https://api.github.com/gists/1c93ded6e78d67b682cabe240b0d8f5a
#owner: https://api.github.com/users/davidel

#!/bin/bash

set -ex

if [ ! -d ".git" ]; then
    echo "Must be run from within a GIT repository!"
    exit 1
fi

MAIN_BRANCH=$(git config --get init.defaultBranch || echo master)
DEV_BRANCH=$(git branch --no-color --show-current)
if [[ $DEV_BRANCH == $MAIN_BRANCH ]]; then
    echo "Cannot squash main branch: $DEV_BRANCH"
    exit 1
fi

read -p "[$DEV_BRANCH] You will be squashing all the GIT branch commits! Are you sure? (y/n)" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting ..."
    exit 1
fi

git reset --soft $(git merge-base $MAIN_BRANCH HEAD)
git commit -m "Changes in $DEV_BRANCH branch."
git push --force origin "$DEV_BRANCH"
