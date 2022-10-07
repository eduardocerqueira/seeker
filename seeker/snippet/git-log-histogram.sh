#date: 2022-10-07T17:25:57Z
#url: https://api.github.com/gists/ae6d2afbcb6715dd4a1d897b8e4104ad
#owner: https://api.github.com/users/chrismwendt

#!/usr/bin/env bash

branch="$1"
buckets="$2"

if [ -z "$buckets" ]; then
  echo "Usage: git-log-histogram <branch> <buckets>"
  echo
  echo "Prints the number of commits on <branch> in each of the last <buckets> seconds/minutes/hours/days."
  echo
  exit 1
fi

echo "Last $buckets seconds:"
env TZ=UTC0 git log "$branch" --quiet --date='format-local:%Y-%m-%d %H:%M:%SZ' --format="%cd" --since="$buckets seconds ago" | uniq -c
echo

echo "Last $buckets minutes:"
env TZ=UTC0 git log "$branch" --quiet --date='format-local:%Y-%m-%d %H:%MZ' --format="%cd" --since="$buckets minutes ago" | uniq -c
echo

echo "Last $buckets hours:"
env TZ=UTC0 git log "$branch" --quiet --date='format-local:%Y-%m-%d %HZ' --format="%cd" --since="$buckets hours ago" | uniq -c
echo

echo "Last $buckets days:"
env TZ=UTC0 git log "$branch" --quiet --date='format-local:%Y-%m-%dZ' --format="%cd" --since="$buckets days ago" | uniq -c
echo
