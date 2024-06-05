#date: 2024-06-05T16:39:25Z
#url: https://api.github.com/gists/9740d7d9960ddeaa8059ad8cfbfa55e0
#owner: https://api.github.com/users/Markkop

#!/bin/bash

# This script automates the creation of a GitHub pull request with information extracted from the current git branch name.
# It expects the branch name to contain a JIRA issue ID followed by a descriptive title.
# The script will:
# 1. Extract the JIRA issue ID from the branch name.
# 2. Format the title from the branch name.
# 3. Prompt the user to classify the issue as a bug or feature.
# 4. Prompt the user to select the target branch for the pull request.
# 5. Create the pull request using the GitHub CLI (`gh`).

# Extract the current branch name
branch_name=$(git rev-parse --abbrev-ref HEAD)

# Extract the JIRA issue ID in lowercase
jira_id_lowercase=$(echo "$branch_name" | grep -o -E '[a-zA-Z]+-[0-9]+' | head -n1)

# Convert the JIRA issue ID to uppercase
jira_id_uppercase=$(echo "$jira_id_lowercase" | tr '[:lower:]' '[:upper:]')

# Format the title by removing the JIRA issue ID and replacing hyphens with spaces
title=$(echo "$branch_name" | sed -E "s/$jira_id_lowercase-([^-]+.*)/\1/g" | sed 's/-/ /g')
title="${title#* }" # Remove the first word (feature/hotfix/etc)

# Check if the JIRA issue ID was extracted successfully
if [[ -z "$jira_id_lowercase" ]]; then
  echo "Could not extract Jira ID from branch name $branch_name"
  exit 1
fi

# Capitalize the first letter of the title
capitalized_title="$(tr '[:lower:]' '[:upper:]' <<< ${title:0:1})${title:1}"

# Display the formatted title to the user
echo The title is going to be: "[$jira_id_uppercase] $capitalized_title"

# Prompt the user to classify the issue as a bug or feature
read -p "Is this issue a bug or a feature? (b/f): " issueType

if [[ $issueType == [Bb]* ]]; then
    label="bug"
elif [[ $issueType == [Ff]* ]]; then
    label="feature"
else
    echo "Invalid input. Please enter 'b' for bug or 'f' for feature."
    exit 1
fi

# Prompt the user to select the target branch for the pull request
read -p "Should it target the main, staging, or dev branch? (m/s/d): " targetBranch

if [[ $targetBranch == [Mm]* ]]; then
    targetBranchName="main"
elif [[ $targetBranch == [Ss]* ]]; then
    targetBranchName="staging"
elif [[ $targetBranch == [Dd]* ]]; then
    targetBranchName="dev"
else
    echo "Invalid input. Please enter 'm' for main, 's' for staging, or 'd' for dev."
    exit 1
fi

# Construct the GitHub CLI command to create the pull request
command="gh pr create -a @me -B $targetBranchName -d -l $label -t \"[$jira_id_uppercase] $capitalized_title\" -b \"https://YOUR_ORG.atlassian.net/browse/$jira_id_uppercase\""
eval $command
