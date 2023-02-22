#date: 2023-02-22T17:02:10Z
#url: https://api.github.com/gists/b650035efa79dd049159252bad07933f
#owner: https://api.github.com/users/Alex-D

#!/usr/bin/env sh

url="example.com"

wget \
  --mirror \
  --no-clobber \
  --page-requisites \
  --html-extension \
  --convert-links \
  --domains $url \
  --no-parent \
  --trust-server-names \
  --execute robots=off \
$url
