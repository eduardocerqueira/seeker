#date: 2023-04-21T16:45:04Z
#url: https://api.github.com/gists/46e53644761bd443032476bddea70772
#owner: https://api.github.com/users/amauryq

#!/usr/bin/env bash

set -u
trap "set +u" EXIT

RED='\033[0;31m'
YELLOW='\033[1;33'
NC='\033[0m' # No Color

for d1 in "${HOME}"/repos/*; do
  for d2 in "$d1"/*; do
    if [ -d "$d2/.git" ]; then
      printf "pulling $d2"
      CURRENT_BRANCH="$(git -C "$d2" rev-parse --abbrev-ref HEAD)"
      [[ "${CURRENT_BRANCH}"=="master" || "${CURRENT_BRANCH}"=="main" ]] || echo -e " => ${YELLOW}WARNING: not on master|main branch${NC}"
      if [ -z "$(git -C "$d2" status --porcelain)" ]; then 
        git -C "$d2" pull --rebase --quiet
        echo " => done"
      else
        echo -e " => ${RED}ERROR: dirty${NC}"
      fi
    fi
  done
done