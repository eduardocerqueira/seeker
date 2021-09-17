#date: 2021-09-17T17:00:47Z
#url: https://api.github.com/gists/ba0cbafbbd85bca2f445cd5fb272589c
#owner: https://api.github.com/users/evanrrees

#!/usr/bin/env bash

# Evan Rees
# err87@cornell.edu
# 13 Sep 2021

# Generate mailto link for joining / leaving Cornell listservs.
# This just wraps the one-liner at the end.
# The link can be opened in any browser or with the macOS `open` command.

declare MAILTO REQUEST

usage() {
  cat <<- EOF
Usage:
  $0 join LIST
  $0 leave LIST
EOF
}

if (( $# == 0 )); then
  usage
  exit 0
elif [[ $1 =~ ^[-]{0,2}h(elp)?$ ]]; then
  usage
  exit 0
elif (( $# != 2 )); then
  echo "Not enough arguments supplied (expected 2, found $#)" > /dev/stderr
  usage
  exit 1
elif ! [[ $2 =~ @ ]]; then
  echo "Invalid argument to LIST: $2" > /dev/stderr
  usage
  exit 1
else
  case "$1" in
    join|j);;
    leave|l);;
    help|-h|--help) usage; exit 0;;
    *)
      echo "Unrecognized argument: $1" > /dev/stderr
      usage
      exit 1
      ;;
  esac
fi

echo "mailto:${2/@/-request@}?subject=$1"
