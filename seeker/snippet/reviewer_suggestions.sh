#date: 2023-03-16T16:50:11Z
#url: https://api.github.com/gists/84e9eaadf1507b2e63626a05004c9429
#owner: https://api.github.com/users/leogdion

#!/bin/bash

tmpfile=$(mktemp)

git diff --name-only develop HEAD | while read -r file; do
	git log -n5 --pretty='format:%an' -- $file >> $tmpfile
	echo "" >> $tmpfile
done

sort $tmpfile | uniq -c | sort -k1,1nr | head -12
