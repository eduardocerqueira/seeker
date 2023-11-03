#date: 2023-11-03T16:58:48Z
#url: https://api.github.com/gists/2b2e0e722c58473e45c7ad7e2b2d84a7
#owner: https://api.github.com/users/rlcamp

#!/bin/sh
set -e

for repo in $(find . -mindepth 2 -type d -name '.git' | sed -e 's/\/.git//'); do
    (cd $repo &&
        git diff --exit-code HEAD &&
        git log --decorate --oneline | head -n1 | grep origin
    ) 1>/dev/null || printf '"%s" has uncommitted or unpushed changes\n' $(basename $repo) >&2
done
