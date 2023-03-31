#date: 2023-03-31T16:48:36Z
#url: https://api.github.com/gists/dfd63783f8c3f5acc79c6f48be76f207
#owner: https://api.github.com/users/Markkop

#!/bin/bash

branch_name=$(git rev-parse --abbrev-ref HEAD)
jira_id_lowercase=$(echo "$branch_name" | grep -o -E '[a-zA-Z]+-[0-9]+' | head -n1)
jira_id_uppercase=$(echo "$jira_id_lowercase" | tr '[:lower:]' '[:upper:]')
title=$(echo "$branch_name" | sed -E "s/$jira_id_lowercase-([^-]+.*)/\1/g" | sed 's/-/ /g')
title="${title#* }" # remove first word (feature/hotfix/etc)


if [[ -z "$jira_id_lowercase" ]]; then
  echo "Could not extract Jira ID from branch name $branch_name"
  exit 1
fi

capitalized_title="$(tr '[:lower:]' '[:upper:]' <<< ${title:0:1})${title:1}"

echo The title is going to be: "[$jira_id_uppercase] $capitalized_title"
# Eg. "[JIRA-123] This is the title" from branch JIRA-123-feature-this-is-the-title

read -p "Is this issue a bug or an enhancement? (b/e): " issueType

if [[ $issueType == [Bb]* ]]; then
    label="bug"
elif [[ $issueType == [Ee]* ]]; then
    label="enhancement"
else
    echo "Invalid input. Please enter 'b' for bug or 'e' for enhancement."
    exit 1
fi

read -p "Should it target the main or the staging branch? (m/s): " targetBranch

if [[ $targetBranch == [Mm]* ]]; then
    targetBranchName="main"
elif [[ $targetBranch == [Ss]* ]]; then
    targetBranchName="staging"
else
    echo "Invalid input. Please enter 'm' for main or 's' for staging."
    exit 1
fi

command="gh pr create -a @me -B $targetBranchName -d -l $label -t \"[$jira_id_uppercase] $capitalized_title\" -b \"YOUR_JIRA_URL/browse/$jira_id_uppercase\""
# Eg: gh pr create -a @me -B staging -d -l enhancement -t "[JIRA-123] This is the title" -b "YOUR_JIRA_URL/browse/JIRA-123"

eval $command

# Eg.
# Mark git:(JIRA-123-feature-this-is-the-title) ./scripts/createPullRequest.sh
# The title is going to be: [JIRA-123] This is the title
# Is this issue a bug or an enhancement? (b/e): e
# Should it target the main or the staging branch? (m/s): s
# 
# Creating draft pull request for JIRA-123-feature-this-is-the-title into staging in halbornlabs/halborn-com
#
# https://github.com/{user}/{project}/pull/1