#date: 2024-01-18T17:08:10Z
#url: https://api.github.com/gists/bb754e33eca36f0b74d4fbe9c7fa4024
#owner: https://api.github.com/users/rydurham

list_all_branches() {
    for k in `git branch -a | perl -pe 's/^..(.*?)( ->.*)?$/\1/'`; do echo -e `git show --pretty=format:"%Cgreen%ci %Cblue%cr%Creset" $k -- | head -n 1`\\t$k; done | sort -r
}

git_purge() {
    # have any arguments been passed?
    if [ $# -gt 0 ];then
        BRANCH=$1
        echo $BRANCH

        # https://stevenharman.net/git-clean-delete-already-merged-branches
        git checkout $BRANCH
        git branch --merged $BRANCH | grep -v "\* $BRANCH" | xargs -n 1 git branch -d
        git remote prune origin
    else
        echo "No branch specified"
    fi
}

alias gs='git status --short --branch'
alias gba=list_all_branches
alias gp=git_purge