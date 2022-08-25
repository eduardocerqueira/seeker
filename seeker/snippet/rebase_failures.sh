#date: 2022-08-25T17:00:17Z
#url: https://api.github.com/gists/6576556282d776a3f66de4203d7f4869
#owner: https://api.github.com/users/carlocab

#!/bin/bash

LIMIT=250
CLOSE_MESSAGE="please reopen against master"
failed_prs="$(gh pr list --limit $LIMIT --search 'status:failure' | cut -f1)"

git fetch origin

for pr in $failed_prs
do
    gh pr checkout "$pr"
    if ! git rebase origin/master
    then
        git rebase --abort
        gh pr close "$pr" --comment "$CLOSE_MESSAGE"
    elif ! git push --force-with-lease
    then
        gh pr close "$pr" --comment "$CLOSE_MESSAGE"
    fi
done