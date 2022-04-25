#date: 2022-04-25T17:09:27Z
#url: https://api.github.com/gists/a762f3a5d200ecc1e2c5f55df8206842
#owner: https://api.github.com/users/bandogora

find . -type f -not -path './db/*' -iregex '.*.\(rb\|haml\|erb\)' -exec sed -ri 's/(\.count)/\.size/g' {} \;