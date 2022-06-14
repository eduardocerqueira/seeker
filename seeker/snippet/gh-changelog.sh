#date: 2022-06-14T16:56:14Z
#url: https://api.github.com/gists/eb5299241349cae5b5c51b7c814cd341
#owner: https://api.github.com/users/cwilper

#!/usr/bin/env bash
#
# Prints a markdown changelog for a GitHub project, including a section for each release, with links to all merged PRs.
#
# Assumes all merged tags represent releases, and the remote named "origin" is the github one.
#
# Usage:
#
# git checkout main ; git pull
# curl -o- https://gist.githubusercontent.com/cwilper/eb5299241349cae5b5c51b7c814cd341/raw/e2819713842509a3a11053940532fc75706048c8/gh-changelog.sh | bash > CHANGELOG.md
#
# By default, all releases are included in the output, but if a tag is specified, output is limited to that release.
#
# If the GH_RELNOTES_TOKEN environment variable is defined, it will be used to make REST requests to the
# GitHub REST API to determine PR titles. Otherwise, titles of each PR link will just be the PR number.
#

iso_commit_date() {
  git show --pretty=fuller --date=iso $1 | grep CommitDate: | awk '{print $2}'
}

pr_title() {
  echo -n "PR #$1"
  if [[ -n $GH_RELNOTES_TOKEN ]]; then
    sleep 0.1
    echo -n " - "
    curl -H "Authorization: token $GH_RELNOTES_TOKEN" https://api.github.com/repos/$project/pulls/$1 2> /dev/null | jq -r .title
  else
    echo
  fi
}

print_for_tag() {
  local tag=$1
  local date=$2
  local prev_ref=$3
  echo "## $tag - $date"
  echo
  echo "**Merged PRs:**"
  echo
  git log --pretty=oneline $prev_ref...$tag \
    | grep "Merge pull request" \
    | awk '{print $5}' \
    | sed 's|#||g' \
    | sort -n | while read -r n; do
      echo "* [$(pr_title $n)](https://github.com/$project/pull/$n)"
  done
  echo
  echo "**Code:**"
  echo
  echo "* [Browse](https://github.com/$project/tree/$tag)"
  echo "* [View diffs](https://github.com/$project/compare/$prev_ref...$tag)"
  echo
}

# determine project from origin url
project=$(git remote -v | grep origin | grep github.com | head -1 | sed 's|.git |/|' | awk -F/ '{print $4"/"$5}')
if [[ -z $project ]]; then
  echo "Error: Cannot determine GitHub project from origin URL"
  exit 1
fi

if [[ -z $1 ]]; then
  echo "# Changelog"
  echo
  echo "<!-- Generated with gh-changelog.sh from https://gist.github.com/cwilper/eb5299241349cae5b5c51b7c814cd341 -->"
  echo
fi

prev_ref=$(git rev-list --max-parents=0 HEAD)
for tag in $(git tag --merged); do
  echo "$(iso_commit_date $tag) $tag"
done | sort | while read -r line; do
  arr=($line)
  echo "$line $prev_ref"
  prev_ref=${arr[1]}
done | sort -r | while read -r line; do
  arr=($line)
  if [[ -z $1 || $1 == ${arr[1]} ]]; then
    print_for_tag ${arr[1]} ${arr[0]} ${arr[2]}
  fi
done