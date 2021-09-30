#date: 2021-09-30T17:15:53Z
#url: https://api.github.com/gists/e8153b36915b1c558dc750ab9d04c0f0
#owner: https://api.github.com/users/halaxa

TOKEN= # Create github token here https://github.com/settings/tokens and check `repo` scope 
USERNAME= # github user name
ISSUE_ID= # issue id without '#'
PR_HEAD= # usually your feature branch
PR_BASE= # usually `master`
REPO= # repository name

curl --user "$USERNAME:$TOKEN" \
  --header "Accept: application/vnd.github.v3+json" \
  --request POST \
  --data "{\"issue\": $ISSUE_ID, \"head\": \"$PR_HEAD\", \"base\": \"$PR_BASE\"}" \
  "https://api.github.com/repos/$USERNAME/$REPO/pulls"
