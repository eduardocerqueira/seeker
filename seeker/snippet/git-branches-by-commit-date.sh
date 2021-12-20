#date: 2021-12-20T17:09:36Z
#url: https://api.github.com/gists/2e3f0c8970ff80081ab527a9690195dd
#owner: https://api.github.com/users/gthrm

# Credit http://stackoverflow.com/a/2514279
for branch in `git branch -r | grep -v HEAD`;do echo -e `git show --format="%ci %cr" $branch | head -n 1` \\t$branch; done | sort -r