#date: 2022-09-16T17:28:15Z
#url: https://api.github.com/gists/717065f6552626fdde7174508f8bb750
#owner: https://api.github.com/users/wbv

#!/bin/sh

# get your `fortune` from a random commit in the current git repository
# falling back to a helpful message

git rev-parse --is-inside-work-tree >/dev/null 2>&1
in_git_repo=$?
if [ $in_git_repo -ne 0 ]; then
	cat <<-EOF
I'm not being run from inside a git repository.

 -- git-fortune.sh
EOF
	exit $in_git_repo
fi

git rev-list --all \
	| cut -f1 -d' ' \
	| sort -R \
	| head -n 1 \
	| git rev-list \
		--no-walk \
		--no-commit-header  \
		--stdin  \
		--format="format:%s%b%n%n -- %an"
