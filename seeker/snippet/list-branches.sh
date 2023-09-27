#date: 2023-09-27T16:46:38Z
#url: https://api.github.com/gists/4cd927cca73c666037dbb5b509d3a91f
#owner: https://api.github.com/users/leighklotz

#!/bin/bash

for branch in $(git branch | sed -e 's/[*]//')
do
  echo -ne "${branch}\tis contained in these branches:\t"
  git branch --contains "${branch}" | tr '\n*' '  ' | sed -e "s/\<${branch}\>//"
  echo
done


