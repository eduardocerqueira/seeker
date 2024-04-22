#date: 2024-04-22T17:03:53Z
#url: https://api.github.com/gists/a27df5c3b70cd4e1fdce62500e38304a
#owner: https://api.github.com/users/andronedev

#!/bin/bash

# Basic Documentation
cat << EOF
GitHub Artifact Cleaner
-----------------------
This script deletes all artifacts from a specified GitHub repository.

USAGE:
1. Ensure that the provided GitHub token has the necessary permissions to access the artifacts.
2. Run the script and follow the on-screen instructions to enter the repository URL and GitHub token.
EOF

# Prompt for repository URL
read -p "Enter the GitHub repository URL: " repo_url
# Prompt for GitHub authentication token
read -p "Enter your GitHub token: "**********"

# Extract owner and repository name from the URL
owner=$(echo $repo_url | sed -E 's|https://github.com/([^/]*)/([^/]*)|\1|')
repo=$(echo $repo_url | sed -E 's|https://github.com/([^/]*)/([^/]*)|\2|')

# Function to fetch all artifacts
function fetch_artifacts {
    echo "Fetching list of artifacts..."
    artifact_ids=$(curl -s -H "Authorization: "**********"
                         -H "Accept: application/vnd.github.v3+json" \
                         "https://api.github.com/repos/$owner/$repo/actions/artifacts" | jq -r '.artifacts[] | .id')

    if [[ -z "$artifact_ids" ]]; then
        echo "No artifacts to delete."
        exit 0
    fi
    echo "$artifact_ids"
}

# Function to delete an artifact
function delete_artifact {
    artifact_id=$1
    echo "Deleting artifact ID $artifact_id..."
    response=$(curl -s -X DELETE -H "Authorization: "**********"
                         -H "Accept: application/vnd.github.v3+json" \
                         "https://api.github.com/repos/$owner/$repo/actions/artifacts/$artifact_id")
    echo "Artifact $artifact_id deleted."
}

# Get all artifacts and delete them one by one
artifact_ids=$(fetch_artifacts)

for id in $artifact_ids; do
    delete_artifact $id
done

echo "All artifacts have been deleted."
s have been deleted."
