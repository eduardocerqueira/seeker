#date: 2026-03-02T17:40:56Z
#url: https://api.github.com/gists/962d0fbe82b1f8aa18dcfc73687c835d
#owner: https://api.github.com/users/liuliu

#!/usr/bin/env bash

set -euo pipefail

mkdir -p _git

BRANCH=$(git --git-dir=$1/.git branch --show-current)

mv $1/.git _git/$1.git

git --git-dir=_git/$1.git config core.bare true

mv $1 $1_tmp

git --git-dir=_git/$1.git worktree add $1 -b $1_tmp

mv $1/.git $1_tmp/.git

rm -rf $1

mv $1_tmp $1

git --git-dir=$1/.git checkout $BRANCH
git --git-dir=$1/.git branch -d $1_tmp