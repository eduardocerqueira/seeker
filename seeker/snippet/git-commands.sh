#date: 2025-03-19T17:01:39Z
#url: https://api.github.com/gists/c744618c1e93383e0469e71c8d6791ae
#owner: https://api.github.com/users/LeonardMeagher2

# Squash all commits in the current branch
git config --global alias.squash-all '!f(){ git reset $(git commit-tree "HEAD^{tree}" "$@");};f'

# Squash all your commits onto the history of another branch
git config --global alias.squash-onto '!f(){ git reset --soft "$1" && git add -A && if [ -n "$2" ]; then git commit -m "$2"; else git commit; fi; }; f'
