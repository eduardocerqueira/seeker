#date: 2025-04-25T17:11:32Z
#url: https://api.github.com/gists/b452a3eaa1096f1ac029bc46d031c859
#owner: https://api.github.com/users/TitusRobyK

#!/bin/bash

# --- Configuration ---
RELEASE_BASE_BRANCH="release/1.1.0"
DEVELOP_BRANCH="develop"
REMOTE_NAME="azure"

# --- Script Logic ---

echo "--- Starting Change Cascade Script ---"

# 1. Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ $? -ne 0 ]; then
    echo "Error: Failed to get current branch name. Are you in a git repository?"
    exit 1
fi
echo "Detected current feature branch: $CURRENT_BRANCH"

# Basic check
if [[ "$CURRENT_BRANCH" == "$DEVELOP_BRANCH" ]] || [[ "$CURRENT_BRANCH" == "$RELEASE_BASE_BRANCH" ]]; then
    echo "Error: This script should be run from a feature branch (e.g., feature/xyz), not directly on '$DEVELOP_BRANCH' or '$RELEASE_BASE_BRANCH'."
    exit 1
fi

# 2. Get current git user's email
CURRENT_USER_EMAIL=$(git config user.email)
if [ -z "$CURRENT_USER_EMAIL" ]; then
    echo "Error: Git user email is not configured. Please set it using:"
    echo "  git config --global user.email \"you@example.com\""
    exit 1
fi
echo "Identifying commits by author: $CURRENT_USER_EMAIL"

# 3. Find commit IDs and Subjects on the current branch since the release base
echo "Finding commits on '$CURRENT_BRANCH' since '$RELEASE_BASE_BRANCH' authored by '$CURRENT_USER_EMAIL'..."

# Ensure the base branch ref is available locally
git fetch $REMOTE_NAME $RELEASE_BASE_BRANCH > /dev/null 2>&1

# Store commit hash and subject, separated by a tab (or other delimiter)
# Use --reverse for chronological order
SOURCE_COMMITS_INFO=$(git log ${REMOTE_NAME}/${RELEASE_BASE_BRANCH}..HEAD --author="$CURRENT_USER_EMAIL" --format="%H%x09%s" --reverse)

if [ $? -ne 0 ]; then
    echo "Error: Failed to run git log. Please check your setup."
    exit 1
fi

if [ -z "$SOURCE_COMMITS_INFO" ]; then
    echo "Warning: No commits found on '$CURRENT_BRANCH' authored by '$CURRENT_USER_EMAIL' since it diverged from '$RELEASE_BASE_BRANCH'."
    exit 1
fi

echo "Found potential commits to cherry-pick (oldest first):"
echo "$SOURCE_COMMITS_INFO" | awk -F'\t' '{print "- " $1 " (" $2 ")"}' # Pretty print hash and subject
echo # Newline for readability

# 4. Define target branch name
NEW_BRANCH_NAME="${CURRENT_BRANCH}-develop"

# 5. Checkout or Create the new develop-based feature branch
BRANCH_EXISTED=false
echo "Checking for target branch '$NEW_BRANCH_NAME'..."
if git checkout "$NEW_BRANCH_NAME" > /dev/null 2>&1; then
    echo "Branch '$NEW_BRANCH_NAME' exists. Checking it out."
    BRANCH_EXISTED=true
    # Ensure local develop ref is up-to-date for comparison purposes
    echo "Fetching latest '$DEVELOP_BRANCH' from '$REMOTE_NAME' to update base reference..."
    git fetch $REMOTE_NAME $DEVELOP_BRANCH:$DEVELOP_BRANCH --update-head-ok > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to fetch latest '$DEVELOP_BRANCH'. Base for comparison might be stale."
    fi
    # Optional: git pull origin $NEW_BRANCH_NAME # Or pull the existing branch itself? Consider implications.
else
    echo "Branch '$NEW_BRANCH_NAME' does not exist. Creating it..."
    # Ensure local develop is up-to-date before creating the new branch
    echo "Checking out '$DEVELOP_BRANCH' to update..."
    git checkout $DEVELOP_BRANCH
    if [ $? -ne 0 ]; then echo "Error: Failed checkout '$DEVELOP_BRANCH'"; exit 1; fi
    echo "Pulling latest changes for '$DEVELOP_BRANCH' from '$REMOTE_NAME'..."
    git pull $REMOTE_NAME $DEVELOP_BRANCH
    if [ $? -ne 0 ]; then echo "Error: Failed pull for '$DEVELOP_BRANCH'"; exit 1; fi

    echo "Creating and checking out new branch '$NEW_BRANCH_NAME'..."
    git checkout -b "$NEW_BRANCH_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create new branch '$NEW_BRANCH_NAME'."
        exit 1
    fi
fi

# 6. Pre-fetch subject lines from target branch for duplicate checking
TARGET_SUBJECT_LINES=""
if [ "$BRANCH_EXISTED" = true ]; then
    echo "Fetching existing commit subjects from '$NEW_BRANCH_NAME' since '$DEVELOP_BRANCH'..."
    # Fetch subjects of commits added to NEW_BRANCH_NAME after diverging from develop
    TARGET_SUBJECT_LINES=$(git log ${REMOTE_NAME}/${DEVELOP_BRANCH}.."$NEW_BRANCH_NAME" --format=%s --)
    if [ $? -ne 0 ]; then
         echo "Warning: Failed to get commit subjects from existing branch '$NEW_BRANCH_NAME'. Duplicate detection will be skipped."
         TARGET_SUBJECT_LINES=""
    fi
    echo "Done fetching subjects."
fi


# 7. Cherry-pick the commits one by one, skipping duplicates based on subject
echo "Starting cherry-pick process onto '$NEW_BRANCH_NAME'..."
PICKED_COUNT=0
SKIPPED_COUNT=0
PROCESSED_COUNT=0
TOTAL_COUNT=$(echo "$SOURCE_COMMITS_INFO" | wc -l | xargs)

# Use Process Substitution or a while loop to read lines containing tabs/spaces correctly
while IFS=$'\t' read -r COMMIT_ID SUBJECT; do
    if [ -z "$COMMIT_ID" ]; then continue; fi # Skip empty lines if any

    ((PROCESSED_COUNT++))
    echo "--------------------------------------------------"
    echo "Processing commit $PROCESSED_COUNT/$TOTAL_COUNT: $COMMIT_ID ('$SUBJECT')"

    # Check if commit subject already exists on the target branch since develop
    ALREADY_PICKED=false
    if [ "$BRANCH_EXISTED" = true ] && [ -n "$TARGET_SUBJECT_LINES" ]; then
        # Use grep -F (fixed string) -x (whole line) -q (quiet) for exact match
        if echo "$TARGET_SUBJECT_LINES" | grep -Fxq -- "$SUBJECT"; then
            ALREADY_PICKED=true
        fi
    fi

    if [ "$ALREADY_PICKED" = true ]; then
        echo "⏭️ Skipping commit $COMMIT_ID - Subject already found on branch '$NEW_BRANCH_NAME'."
        ((SKIPPED_COUNT++))
        continue # Move to the next commit
    fi

    # If not skipped, attempt cherry-pick
    echo "Attempting cherry-pick for $COMMIT_ID..."
    git cherry-pick "$COMMIT_ID"

    # Check for conflicts
    if [ $? -ne 0 ]; then
        echo "-------------------------------------------------------------"
        echo "⛔ CONFLICT DETECTED while cherry-picking commit $COMMIT_ID. ⛔"
        echo "-------------------------------------------------------------"
        echo "The script has paused."
        echo "Please resolve the conflicts in your editor/tool."
        echo "Once resolved, stage the changes using:"
        echo "  git add <resolved_file_1> <resolved_file_2> ..."
        echo "Then, continue the cherry-pick process with:"
        echo "  git cherry-pick --continue"
        echo ""
        echo "Alternatively, to skip this problematic commit:"
        echo "  git cherry-pick --skip"
        echo "Or to cancel the entire cherry-pick operation:"
        echo "  git cherry-pick --abort"
        echo ""
        echo "After resolving and continuing/skipping, you may need to manually cherry-pick any remaining commits listed above."
        echo "-------------------------------------------------------------"
        # Exit the script - user interaction is required.
        exit 1
    else
        echo "✅ Successfully cherry-picked $COMMIT_ID."
        ((PICKED_COUNT++))
        # Add the subject to our list to avoid potential duplicate picks within the same run
        # (e.g., if the source branch had two identical commit messages)
         if [ -n "$TARGET_SUBJECT_LINES" ]; then
             TARGET_SUBJECT_LINES+=$'\n'"$SUBJECT"
         else
             TARGET_SUBJECT_LINES="$SUBJECT"
         fi
    fi

done <<< "$SOURCE_COMMITS_INFO" # Feed the commit info into the while loop


# --- Final Summary ---
echo "--------------------------------------------------"
echo "✅ Cherry-pick process finished for '$NEW_BRANCH_NAME'."
echo "Current branch is now '$NEW_BRANCH_NAME'."
echo "Summary:"
echo "  - Commits successfully cherry-picked: $PICKED_COUNT"
echo "  - Commits skipped (subject found):  $SKIPPED_COUNT"
echo "  - Total commits processed:          $PROCESSED_COUNT" # Should match TOTAL_COUNT if no errors
echo ""
echo "Please review the changes carefully (e.g., using 'git log' or 'gitk')."
echo "If everything looks correct, you can push the branch:"
echo "  git push $REMOTE_NAME $NEW_BRANCH_NAME"
echo "--- Change Cascade Script Finished ---"

exit 0