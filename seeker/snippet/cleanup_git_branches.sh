#date: 2022-03-10T17:13:51Z
#url: https://api.github.com/gists/d134971e55e4f0cbdeae2ee1886ffe36
#owner: https://api.github.com/users/krcm0209

#!/bin/sh
git fetch -p
branches_with_gone_remote=$(
git for-each-ref --format '%(refname) %(upstream:track)' refs/heads |
	awk '$2 == "[gone]" {sub("refs/heads/", "", $1); print $1}'
)

for branch in ${branches_with_gone_remote}
do
	git branch -D $branch
done