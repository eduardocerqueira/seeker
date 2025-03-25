#date: 2025-03-25T17:05:29Z
#url: https://api.github.com/gists/68f1a49fa81e846c221464380aa81d00
#owner: https://api.github.com/users/melizeche

git log --author=<emailregex> --pretty=tformat: --numstat | awk '{ adds += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s, added:deleted ratio:%s\n", adds, subs, loc, adds/subs }' -