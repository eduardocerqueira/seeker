#date: 2025-10-14T17:06:55Z
#url: https://api.github.com/gists/772fee110bb1c0a6d98593f8e5ac57e7
#owner: https://api.github.com/users/inxilpro

#!/usr/bin/env bash

# Move into project root using git
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit

# Configuration
INCLUDE_COMMIT_HISTORY=true
COMMIT_HISTORY_COUNT=5

# Get commit message file path
COMMIT_MSG_FILE=$1

if [[ ! -f "$COMMIT_MSG_FILE" ]]; then
  echo "Error: Commit message file not found"
  exit 1
fi

# Read the commit message
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Trim whitespace
COMMIT_MSG=$(echo "$COMMIT_MSG" | xargs)

# Convert to lowercase for case-insensitive comparison
COMMIT_MSG_LOWER=$(echo "$COMMIT_MSG" | tr '[:upper:]' '[:lower:]')

# Check if message needs generation
SHOULD_GENERATE=false

# Check if empty
if [[ -z "$COMMIT_MSG" ]]; then
  SHOULD_GENERATE=true
fi

# Check if WIP (case-insensitive)
if [[ "$COMMIT_MSG_LOWER" == "wip" ]]; then
  SHOULD_GENERATE=true
fi

# Check if it's "Update [filename]" for a single-file change
if [[ ! $SHOULD_GENERATE == true ]]; then
  # Get list of changed files
  mapfile -t CHANGED_FILES < <(git diff --cached --name-only)

  # If only one file changed
  if [[ ${#CHANGED_FILES[@]} -eq 1 ]]; then
    FILE_BASENAME=$(basename "${CHANGED_FILES[0]}")
    EXPECTED_MSG="update $FILE_BASENAME"
    EXPECTED_MSG_LOWER=$(echo "$EXPECTED_MSG" | tr '[:upper:]' '[:lower:]')

    if [[ "$COMMIT_MSG_LOWER" == "$EXPECTED_MSG_LOWER" ]]; then
      SHOULD_GENERATE=true
    fi
  fi
fi

# If message is fine, exit
if [[ ! $SHOULD_GENERATE == true ]]; then
  exit 0
fi

# Check if Claude CLI is available
if ! command -v claude &> /dev/null; then
  # Claude not available, replace empty message with WIP
  if [[ -z "$COMMIT_MSG" ]]; then
    echo "WIP" > "$COMMIT_MSG_FILE"
  fi
  exit 0
fi

# Get staged changes
DIFF=$(git diff --cached)

if [[ -z "$DIFF" ]]; then
  # No changes, use WIP
  echo "WIP" > "$COMMIT_MSG_FILE"
  exit 0
fi

# Build the prompt
PROMPT="Generate a concise commit message for these changes. Respond with only the commit message, no additional text or formatting. Focus on what other developers
would need to know about the commit. Do not include qualitative statements or speculate as to why the changes were made.

Staged changes:
\`\`\`
$DIFF
\`\`\`"

# Include commit history if configured
if [[ "$INCLUDE_COMMIT_HISTORY" = true ]]; then
  HISTORY=$(git log -n "$COMMIT_HISTORY_COUNT" --format='%s')
  PROMPT="$PROMPT

Recent commit messages for style reference:
\`\`\`
$HISTORY
\`\`\`"
fi

# Call Claude to generate commit message
GENERATED_MSG=$(echo "$PROMPT" | claude 2>&1)
CLAUDE_EXIT_CODE=$?

# Check if Claude succeeded
if [[ $CLAUDE_EXIT_CODE -eq 0 && -n "$GENERATED_MSG" ]]; then
  # Write generated message to commit file
  echo "$GENERATED_MSG" > "$COMMIT_MSG_FILE"
else
  # Claude failed, use WIP
  echo "WIP" > "$COMMIT_MSG_FILE"
fi

exit 0
