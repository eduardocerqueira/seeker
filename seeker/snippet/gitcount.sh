#date: 2022-07-13T16:55:25Z
#url: https://api.github.com/gists/6999702b2da7902e50d642e511989dcf
#owner: https://api.github.com/users/burrsettles

git ls-files -z | xargs -0n1 git blame -w | perl -n -e '/^.*?\((.*?)\s+[\d]{4}/; print $1,"\n"' | sort -f | uniq -c | sort -rnk 1  