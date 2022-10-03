#date: 2022-10-03T17:35:20Z
#url: https://api.github.com/gists/ff83d81e09e5e030f817cf6212c6951f
#owner: https://api.github.com/users/eculver

#!/bin/bash
# changelog_checker.sh
#
# example:
#   ./changelog_checker.sh release/1.11.x main 14828



set -euo pipefail

function main {
  local base_branch
  local default_branch
  local pr_number

  base_ref=${1:-}
  default_branch=${2:-}
  pr_number=${3:-}

  # the result will be 1 when there is equality (when the base branch is main), 0 otherwise
  # pull_request_base_main=$(expr "${base_ref}" = "${default_branch")

  # check if there is a diff in the .changelog directory
  # for PRs against the main branch, the changelog file name should match the PR number
  # as noted above, when ${pull_request_base_main} == 1 the base branch is "main"
  echo "checking base"
  if [ "${base_ref}" = "main" ]; then
    echo "base is main"
    enforce_matching_pull_request_number="matching this PR number "
    changelog_file_path=".changelog/${pr_number}.txt"
  else
    echo "base is NOT main"
    changelog_file_path=".changelog/*.txt"
  fi

  echo "getting changed changelog files"
  changelog_files=$(git --no-pager diff --name-only HEAD "$(git merge-base HEAD "origin/main")" -- ${changelog_file_path})

  # If we do not find a file in .changelog/, we fail the check
  if [ -z "$changelog_files" ]; then
    # Fail status check when no .changelog entry was found on the PR
    echo "Did not find a .changelog entry ${enforce_matching_pull_request_number}and the 'pr/no-changelog' label was not applied. Reference - https://github.com/hashicorp/consul/pull/8387"
    exit 1
  else
    echo "Found .changelog entry in PR!"
  fi
}

main "$@"
