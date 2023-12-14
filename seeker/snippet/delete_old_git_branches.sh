#date: 2023-12-14T16:54:34Z
#url: https://api.github.com/gists/21144563c9820b2cc6d1b19b8729cd6d
#owner: https://api.github.com/users/dharness

git for-each-ref --sort=-committerdate refs/heads --format='%(committerdate:short) %(refname:short)' | awk '$1 < "'$(date -v-2w +%Y-%m-%d)'" {print $2}' | xargs git branch -D