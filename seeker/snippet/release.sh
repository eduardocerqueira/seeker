#date: 2022-07-04T03:23:51Z
#url: https://api.github.com/gists/926ea775aba00c1c3674853ec8d4b500
#owner: https://api.github.com/users/OrchidAugur

#! /usr/bin/env bash

git fetch
git checkout main
git pull
git checkout develop
git pull
git flow release start $1
npm --no-git-tag-version version $1
git commit -am "chore: Bump version to $1"
GIT_MERGE_AUTOEDIT=no git flow release finish -m "$1" $1
git push origin main develop --tags
