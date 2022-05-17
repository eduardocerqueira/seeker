#date: 2022-05-17T17:12:21Z
#url: https://api.github.com/gists/f560980c160422350329fdcf07fed336
#owner: https://api.github.com/users/LiviaMedeiros

#!/bin/bash
git commit --amend --no-edit -S || exit 1
[[ -n "${1}" ]] && git show-ref --quiet refs/heads/"${1}" && git push --force-with-lease origin HEAD:"${1}"
echo "Landed in $(git rev-parse --short=9 HEAD)"

read -rn1 -p "Push to upstream/master? "
echo
[[ ${REPLY} =~ ^[Yy]$ ]] && git push upstream master