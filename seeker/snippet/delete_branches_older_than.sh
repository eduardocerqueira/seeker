#date: 2021-08-31T03:03:31Z
#url: https://api.github.com/gists/f8d54a620af9d8d1e519108d2bedc84f
#owner: https://api.github.com/users/d4rkd0s

#!/bin/sh

ECHO='echo '
for branch in $(git branch -a | sed 's/^\s*//' | sed 's/^remotes\///' | grep -v 'master$\|release$'); do
  if ! ( [[ -f "$branch" ]] || [[ -d "$branch" ]] ) && [[ "$(git log $branch --since "1 month ago" | wc -l)" -eq 0 ]]; then
    if [[ "$DRY_RUN" = "false" ]]; then
      ECHO=""
    fi
    local_branch_name=$(echo "$branch" | sed 's/remotes\/origin\///')
    $ECHO git branch -d "${local_branch_name}"
    $ECHO git push origin --delete "${local_branch_name}"
  fi
done
