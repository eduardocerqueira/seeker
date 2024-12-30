#date: 2024-12-30T16:40:33Z
#url: https://api.github.com/gists/2a4b866846c65f4bd5794ef91651c01f
#owner: https://api.github.com/users/RodAlc24

#!/usr/bin/env bash

# Make sure you have dependencies installed (GitHub cli, GNU parallel)

# Target directory
GITHUB_MIRROR_PATH="$HOME/gh_mirror"

if [[ ! -d $GITHUB_MIRROR_PATH ]]; then 
  mkdir -p $GITHUB_MIRROR_PATH
fi

# Get repos from GitHub
repos=$(gh repo list | awk '{ print $1 }' )

# Go to target directory
pushd $GITHUB_MIRROR_PATH > /dev/null

clone_or_pull() {
  repo=$1
  name=${repo#*/}

  if [[ -d $name ]]; then
    pushd $name > /dev/null
    echo -e "Updating $repo"
    git pull > /dev/null

  else
    echo -e "Cloning $repo"
    gh repo clone $repo > /dev/null

  fi
}

# Needed for parallel
export -f clone_or_pull

# Execute the function using GNU parallel
echo "$repos" | parallel clone_or_pull

