#date: 2021-12-10T17:10:09Z
#url: https://api.github.com/gists/decbaadee21a165bf0bde84131d6a6b0
#owner: https://api.github.com/users/hexium310

git revert $(git log --oneline | nl >&2; read -p '> ' num; git log --format='%h' --skip=$((num - 1)) -1)
