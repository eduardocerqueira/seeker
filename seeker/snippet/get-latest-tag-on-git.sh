#date: 2022-11-15T17:09:48Z
#url: https://api.github.com/gists/600ce8bffe1e504857ca37ba4acc4f24
#owner: https://api.github.com/users/sumanmaity112

#!/usr/bin/env bash

# git version 2.38.1

_get_latest_tag(){
  local pattern="${1}"
  git tag -i -l "${pattern}" --sort -v:refname | head -n1
}

_get_latest_tag "$@"

# ./get-latest-tag-on-git.sh "v*"