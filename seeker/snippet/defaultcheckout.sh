#date: 2021-12-31T16:31:05Z
#url: https://api.github.com/gists/d1e9b53d2f08698925427526cd5462a9
#owner: https://api.github.com/users/CurryEleison

#!/bin/bash

default_branch() {
	git remote show origin | sed -n '/HEAD branch/s/.*: //p' | sed 's/[^a-z0-9_]*//ig'
}

for r in $(ls -1d */)
do
	if [ -d "$r/.git" ]
	then
		pushd "$r"
		git checkout $(default_branch)
		git pull
		popd
	fi
done
