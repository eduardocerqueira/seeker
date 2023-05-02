#date: 2023-05-02T16:57:20Z
#url: https://api.github.com/gists/0c0fa94a101efe795bbc74079c637844
#owner: https://api.github.com/users/longtth

git ls-files | while read f; do git blame --line-porcelain $f | grep '^author '; done | sort -f | uniq -ic | sort -n

# this doesn't include xargs 