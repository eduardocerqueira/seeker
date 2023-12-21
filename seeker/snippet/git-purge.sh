#date: 2023-12-21T16:56:59Z
#url: https://api.github.com/gists/61e1ae3ccf353524077d696cb86f8245
#owner: https://api.github.com/users/rydurham


# https://stevenharman.net/git-clean-delete-already-merged-branches
git_purge() {
    # have any arguments been passed?
    if [ $# -gt 0 ];then
        BRANCH=$1
        echo $BRANCH

        git checkout $BRANCH
        git branch --merged $BRANCH | grep -v "\* $BRANCH" | xargs -n 1 git branch -d
        git remote prune origin
    else
        echo "No branch specified"
    fi
}

alias gp=git_purge