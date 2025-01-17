#date: 2025-01-17T16:58:37Z
#url: https://api.github.com/gists/7a67f1164cf7feb7696b00c2b9e01a90
#owner: https://api.github.com/users/ScottJWalter

#!/bin/sh

# add to .bashrc/.zshrc/... to wrap the cd command with a venv check
# NOTE:  This assumes '.venv' is the virtual environment directory
#
function cd() {
  builtin cd "$@"

  if [[ -z "$VIRTUAL_ENV" ]] ; then
    ## If env folder is found then activate the vitualenv
      if [[ -d ./.venv ]] ; then
        source ./.venv/bin/activate
      fi
  else
    ## check the current folder belong to earlier VIRTUAL_ENV folder
    # if yes then do nothing
    # else deactivate
      parentdir="$(dirname "$VIRTUAL_ENV")"
      if [[ "$PWD"/ != "$parentdir"/* ]] ; then
        deactivate
      fi
  fi
}
