#date: 2025-05-20T16:49:54Z
#url: https://api.github.com/gists/c184cc30804f1d79474782a991a47f4e
#owner: https://api.github.com/users/d-oit

#!/bin/bash
# git commit with codestral

# Exit on any error
set -e

# Function to handle errors
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
for tool in git curl jq; do
    if ! command_exists "$tool"; then
        error_exit "$tool is not installed. Please install it and try again."
    fi
done

# Get the project root directory (one level up from scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root directory: $PROJECT_ROOT"

# Change to project root directory
cd "$PROJECT_ROOT" || error_exit "Failed to change to project root directory"

# Debug: Check for .env file in project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo ".env file exists in project root"
    # Try to source .env file
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "Sourced .env file"
else
    echo ".env file not found in project root"
fi

# Debug: Print API key length (without exposing the key)
echo "API key length: ${#CODESTRAL_API_KEY}"

# Check for API key
if [[ -z "$CODESTRAL_API_KEY" ]]; then
    error_exit "CODESTRAL_API_KEY environment variable is not set. Please set it in your environment or in a .env file."
fi

# Debug: Show current working directory
echo "Working directory: $(pwd)"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    error_exit "Not a git repository. Please run this script from within a git repository."
fi

# Stage all changes
echo "Staging all changes..."
INITIAL_STATUS=$(git status --porcelain)
if [[ -z "$INITIAL_STATUS" ]]; then
    error_exit "No changes to commit. Working directory clean."
fi

# Stage all tracked files with changes
if ! git add -u; then
    error_exit "Failed to stage changes for tracked files. Check your git status and try again."
fi

# Optionally stage new files if user confirms
if git status --porcelain | grep -q "^??"; then
    echo -e "\nUntracked files found:"
    git status --porcelain | grep "^??" | sed 's/^?? /  /'
    read -p "Stage new files? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        if ! git add .; then
            error_exit "Failed to stage new files. Check your git status and try again."
        fi
    fi
fi

# Get both staged and unstaged changes
STAGED_DIFF=$(git diff --cached)
UNSTAGED_DIFF=$(git diff)

# Check if there are any changes at all
if [[ -z "$STAGED_DIFF" && -z "$UNSTAGED_DIFF" ]]; then
    echo "No changes to commit."
    exit 0
fi

# If there are still unstaged changes in tracked files, confirm with user
REMAINING_TRACKED=$(git diff --name-only)
if [[ -n "$REMAINING_TRACKED" ]]; then
    echo -e "\nRemaining unstaged changes in tracked files:"
    echo "$REMAINING_TRACKED" | sed 's/^/  /'
    read -p "Stage these changes? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        if ! git add -u; then
            error_exit "Failed to stage remaining changes."
        fi
        STAGED_DIFF=$(git diff --cached)
    fi
fi

# Truncate diff if too large (Codestral has context limits)
MAX_CHARS=6000
if [[ ${#STAGED_DIFF} -gt $MAX_CHARS ]]; then
    STAGED_DIFF="${STAGED_DIFF:0:$MAX_CHARS}"
    STAGED_DIFF="$STAGED_DIFF\n\n[Diff truncated]"
fi

# Prompt for Codestral
PROMPT="Write a concise and clear Git commit message describing the following staged code changes:\n\n$STAGED_DIFF"

# Create properly escaped JSON payload using jq
JSON_PAYLOAD=$(jq -n \
    --arg model "mistral-medium-latest" \
    --arg content "$PROMPT" \
    '{
        model: $model,
        messages: [{
            role: "user",
            content: $content
        }],
        temperature: 0.1
    }')

# Debug: Print the request payload
echo "Request payload:"
echo "$JSON_PAYLOAD"

# Query Mistral API
echo "Generating commit message..."
RESPONSE=$(curl -s -f -S -w "\n%{http_code}" https://api.mistral.ai/v1/chat/completions \
  -H "Authorization: Bearer $CODESTRAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD") || {
    echo "API Response: $RESPONSE"
    error_exit "Failed to connect to Mistral API. Check your internet connection and API key."
}

# Extract HTTP status code and response body
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')

# Debug: Print API response details
echo "HTTP Status Code: $HTTP_CODE"
echo "Response Body: $RESPONSE_BODY"

# Check HTTP status
if [[ "$HTTP_CODE" != "200" ]]; then
    error_msg=$(echo "$RESPONSE_BODY" | jq -r '.error.message // "Unknown error"' 2>/dev/null)
    error_exit "API request failed with status $HTTP_CODE: ${error_msg:-'Unknown error'}"
fi

# Extract commit message
COMMIT_MSG=$(echo "$RESPONSE_BODY" | jq -r '.choices[0].message.content // .choices[0].content // empty' 2>/dev/null)

if [[ -z "$COMMIT_MSG" ]]; then
    error_exit "Failed to generate commit message. Invalid API response."
fi

# Clean up the commit message
# Extract just the commit message content without markdown formatting
COMMIT_MSG=$(echo "$COMMIT_MSG" | sed -n '/^```/,/^```/ {/^```/!p}' | sed 's/^Here.*changes://' | sed '/^$/d' | sed 's/^[[:space:]]*//')

# Show and confirm
echo -e "\nAI-generated commit message:\n"
echo "$COMMIT_MSG"
echo
read -p "Use this commit message? [Y/n] " -n 1 -r
CONFIRM=${REPLY:-Y}
echo

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    if git commit -m "$COMMIT_MSG"; then
        echo "Commit created successfully."
    else
        error_exit "Failed to create commit. Check your git configuration and try again."
    fi
else
    echo "Commit aborted."
    exit 0
fi
