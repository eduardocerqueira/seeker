#date: 2022-08-25T17:14:56Z
#url: https://api.github.com/gists/308319671a5250dc0bd0766f46670328
#owner: https://api.github.com/users/harini-ua

#!/bin/sh

git filter-branch -f --env-filter "
    GIT_AUTHOR_NAME='Newname'
    GIT_AUTHOR_EMAIL='new@email'
    GIT_COMMITTER_NAME='Newname'
    GIT_COMMITTER_EMAIL='new@email'
  " HEAD