#date: 2022-12-13T16:46:54Z
#url: https://api.github.com/gists/a309317cde70c264e01fc65049f10348
#owner: https://api.github.com/users/arbakker

#!/usr/bin/env bash
#  npm install --save-dev markdown-link-check -g  

total=0
broken=0
rm -f broken-links.log
for file in $(find . -type f -name "*.md" | grep -v ".gitea")
do
	file=$(realpath $file)
	logging=$(markdown-link-check --verbose  "$file" 2>&1)
	if [[ $? -eq 1 ]] ;then
		echo "$logging" >> broken-links.log
		echo "BROKEN LINK(S) IN: ${file}"
		broken=$((broken+1))
	fi
	total=$((total+1))
done

echo "broken links in ${broken}/${total} markdown files"
