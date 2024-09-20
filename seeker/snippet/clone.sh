#date: 2024-09-20T17:10:46Z
#url: https://api.github.com/gists/bb0499e1272f312b82497031d28e91f2
#owner: https://api.github.com/users/alifeee

#!/bin/bash
# quickly clone a GitHub repository
# 1. take user input of a GitHub repository
# 2. attempt to pattern match to an actual repository
# 3. attempt to clone it
# 4. open in file explorer or code editor
# made by alifeee
# version 0.1.0

BASE="/home/alifeee/git"

echo "will clone to ${BASE}"
read -p "repository: " SSH_LOC

echo " input: ${SSH_LOC}"

# e.g., git@github.com:alifeee/blog.git
PAT_FULL="^git@github\.com:([^\/]*)\/([^\/]*).git$"
# e.g., git@github.com:alifeee/blog
PAT_WITHOUT_EXTENSION="^git@github\.com:([^\/]*)\/([^\/]*)$"
# e.g., alifeee/blog
PAT_OWNER_REPO="^([^\/]*)\/([^\/]*)$"
# e.g., blog (only works when specifying default github account)
PAT_REPO="^([^\/]*)$"
DEFAULT_GITHUB_ACCOUNT="alifeee"

if [[ "${SSH_LOC}" =~ $PAT_FULL ]]; then
  echo " match git@github.com:name/repo.git, cloning as-is"
  cloneme="${SSH_LOC}"
elif [[ "${SSH_LOC}" =~ $PAT_WITHOUT_EXTENSION ]]; then
  echo " match git@github.com:name/repo, adding .git"
  cloneme="${SSH_LOC}.git"
elif [[ "${SSH_LOC}" =~ $PAT_OWNER_REPO ]]; then
  echo " match name/repo, adding github and .git"
  cloneme="git@github.com:${SSH_LOC}.git"
elif [[ "${SSH_LOC}" =~ $PAT_REPO ]]; then
  echo " match repo, attempting alifeee"
  cloneme="git@github.com:${DEFAULT_GITHUB_ACCOUNT}/${SSH_LOC}.git"
else
  read -s -n1 -p "no match type found :("
  exit 1
fi

echo " attempting to clone ${cloneme}"

(cd "${BASE}"; git clone "${cloneme}")

folder=$(echo "${cloneme}" | pcregrep -o2 "${PAT_FULL}")
fullpath="${BASE}/${folder}"

if [ ! -d "${fullpath}" ]; then
  read -s -n1 -p " no cloned folder found. curious... press ENTER to exit"
  exit 1
else
  openme="${BASE}/${folder}"
fi

echo "cloned to ${openme}. what now?"
read -p "ENTER to explore, \"code\" to open in VSCodium: " ACTION

if [ "${ACTION}" == "code" ]; then
  codium "${openme}"
else
  # open in explorer
  xdg-open "${openme}"
fi