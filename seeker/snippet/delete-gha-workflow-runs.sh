#date: 2024-12-11T17:11:02Z
#url: https://api.github.com/gists/a3f193e93155d320e1a3c001cc4e43b5
#owner: https://api.github.com/users/clarkritchie

#!/usr/bin/env bash

OWNER=clarkritchie
REPO=${1:-my-stuff} 
WORKFLOW_NAME=${2} # this is the filename, e.g. my-foo-gha.yaml

if [ -z "$WORKFLOW_ID" ]; then
    echo "Must specify a workflow name"
    exit 1
fi

# gh config set pager cat
# GH_PAGER=cat

cat <<EOF
You are about to delete the workflow run history for:
- Owner: $OWNER
- Repository: $REPO
- Workflow name: $WORKFLOW_NAME
EOF

read -p "Do you want to proceed deleting workflow runs? (yes/no): " confirmation

if [[ "$confirmation" == "yes" ]]; then
    gh api repos/$OWNER/$REPO/actions/workflows/$WORKFLOW_NAME/runs --paginate -q '.workflow_runs[] | select(.head_branch != "master") | "\(.id)"' | \
    xargs -n1 -I % gh api repos/$OWNER/$REPO/actions/runs/% -X DELETE
else
    echo "Operation cancelled"
fi