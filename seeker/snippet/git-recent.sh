#date: 2023-12-15T17:03:12Z
#url: https://api.github.com/gists/e34a157ff080484bc62876b89c74b83e
#owner: https://api.github.com/users/inimicus

#!/usr/bin/env bash

# Installation:
# Place anywhere in your $PATH as git-recent and chmod +x
# Then use with `git recent` anywhere you use git
#
# Requirements:
# - gum
# - git (obviously)
#
# Usage:
# git recent [num]
#
# Examples:
# - git recent 5
# - git recent 15
# - git recent 50

function getRecentBranches() {
	git reflog |
		grep -Eio "moving from ([^[:space:]]+)" |
		awk '{ print $3 }' |
		awk ' !x[$0]++' |
		grep -Ev '^[a-f0-9]{40}$' |
		head -n "${1}"
}

function selectBranch() {
	# word splitting is desired
	# shellcheck disable=SC2046
	gum filter --limit=1 $(getRecentBranches "${1}")
}

numRecent="${1:-15}"

if ! [[ ${numRecent} =~ ^[0-9]+$ ]]; then
	echo "Invalid argument: Number of recent branches should be a number!" >&2
	exit 1
fi

branch="$(selectBranch "${numRecent}")"

if [[ -n "${branch}" ]]; then
	git checkout "${branch}"
fi

exit 0
