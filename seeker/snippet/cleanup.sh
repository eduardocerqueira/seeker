#date: 2022-01-20T16:55:17Z
#url: https://api.github.com/gists/e1aeb72384fce138caac76ecb1500520
#owner: https://api.github.com/users/sergiomasellis

#! /bin/bash
# Deletes remote git branches which are older than 6 weeks and have been merged
#
# Author: Matt Foster <mpf@hackerific.net>

function process_args {
  while getopts ":hlw:" opt; do
    case $opt in
      w)
        WEEKS=$OPTARG
        ;;
      l)
        LIVE=1
        ;;
      h)
        usage
        exit 1
        ;;
      :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        exit 1
        ;;
    esac
  done

  WEEKS=${WEEKS:-6}
  LIVE=${LIVE:-0}

}

function usage {
  echo "Usage: purge-branches [-w <weeks>] [-l]" >&2
  echo "   -w <weeks> - remove merged brances whose latest commit is as least <weeks> weeks old" >&2
  echo "   -l         - actually remove the branches instead of just showing what to run" >&2
  echo "   -h         - show this help" >&2
}

function get_limit {
  unamestr=$(uname)
  if [[ "$unamestr" == 'Darwin' ]]; then
    LIMIT=$(date -j -v-${WEEKS}w +%s)
  else
    LIMIT=$(date --date="$WEEKS weeks ago" +%s)
  fi
}

function get_branches {
  git for-each-ref --sort=-committerdate refs/remotes --format="%(refname) %(committerdate:raw)"
}

function filter_by_date {
  while read branch date zone; do 
    if [[ "$date" -le "$LIMIT" ]]; then
      echo $branch
    fi

  done
}

function clean_branch_name {
  sed -e 's~^\s\+~~'              \
  | sed -e 's~refs/~~'            \
  | sed -e 's~remotes/origin/~~'  \
  | sed -e 's~^origin/~~'
}

function merged_branches {
  git branch -r --merged
}

function delete_branches {
  while read branch; do
    if [[ "$LIVE" -eq "1" ]]; then
      git push origin --delete $branch
    else 
      echo git push origin --delete $branch
    fi
  done
}


# Here's the main program!

process_args "$@"

get_limit

git fetch --all

old_branches=$(mktemp /tmp/purge-branches.XXXXXX)
merged_branches=$(mktemp /tmp/purge-branches.XXXXXX)

get_branches \
  | filter_by_date \
  | clean_branch_name \
  | sort \
  > $old_branches

merged_branches | fgrep -v ' -> ' | fgrep -v 'master' \
  | clean_branch_name \
  | sort \
  > $merged_branches

join $old_branches $merged_branches \
  | delete_branches