#date: 2024-07-18T16:41:42Z
#url: https://api.github.com/gists/ac4fc22d4d4c3b950646f26d7e05a6b7
#owner: https://api.github.com/users/oaliuBC

#!/bin/bash

PERMISSION="push" # Can be one of: pull, push, admin, maintain, triage
ORG="orgname"
TEAM_SLUG="your-team-slug"

# Get names with `gh repo list orgname`
REPOS=(
  "orgname/reponame"
)

for REPO in "${REPOS[@]}"; do
  echo "Adding repo ${REPO} to Org:$ORG Team:$TEAM_SLUG"

  # https://docs.github.com/en/rest/teams/teams#add-or-update-team-repository-permissions
  # (needs admin:org scope)
  # --silent added to make it less noisy
  gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    --silent \
    "/orgs/$ORG/teams/$TEAM_SLUG/repos/$REPO" \
    -f permission="$PERMISSION" && echo 'Added' || echo 'Failed'

  echo "\n============================================================\n"
done
