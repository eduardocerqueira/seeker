#date: 2025-08-07T17:01:12Z
#url: https://api.github.com/gists/3b6dcfd442ca1293a8159f2d0f516220
#owner: https://api.github.com/users/bryangingechen

#!/bin/bash

# Script to filter GitHub workflow runs where a specific job failed
# Usage: ./filter_failed_runs.sh [OPTIONS]
# requires gh and jq to be installed; also requires GNU date via homebrew on macOS

set -e

# Detect the correct date command to use
if command -v gdate &> /dev/null; then
  DATE_CMD="gdate"  # GNU date on macOS (via brew install coreutils)
elif date -d "2024-01-01" &> /dev/null 2>&1; then
  DATE_CMD="date"   # GNU date on Linux
else
  echo "Error: GNU date is required but not found."
  echo "On macOS: brew install coreutils"
  echo "On Linux: date should work by default"
  exit 1
fi

# Default values
WORKFLOW_NAME=""
JOB_NAME=""
LIMIT=10
AFTER_DATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -w|--workflow)
      WORKFLOW_NAME="$2"
      shift 2
      ;;
    -j|--job)
      JOB_NAME="$2"
      shift 2
      ;;
    -l|--limit)
      LIMIT="$2"
      shift 2
      ;;
    -a|--after)
      AFTER_DATE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  -w, --workflow WORKFLOW_NAME    Filter by workflow name (optional)"
      echo "  -j, --job JOB_NAME             Job name to check for failures (required)"
      echo "  -l, --limit NUMBER             Number of workflow runs to check (default: 10)"
      echo "  -a, --after DATE               Only include runs after this date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)"
      echo "  -h, --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if job name is provided
if [[ -z "$JOB_NAME" ]]; then
  echo "Error: Job name is required. Use -j or --job to specify it."
  echo "Use -h or --help for usage information."
  exit 1
fi

# Validate and convert date format if provided
if [[ -n "$AFTER_DATE" ]]; then
  # Try to parse the date to validate it
  if ! $DATE_CMD -d "$AFTER_DATE" &> /dev/null; then
    echo "Error: Invalid date format '$AFTER_DATE'. Use YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'"
    exit 1
  fi
  # Convert to epoch for comparison
  AFTER_EPOCH=$($DATE_CMD -d "$AFTER_DATE" +%s)
fi

# Check if gh is installed and authenticated
if ! command -v gh &> /dev/null; then
  echo "Error: gh CLI is not installed. Please install it first."
  exit 1
fi

if ! gh auth status &> /dev/null; then
  echo "Error: Not authenticated with GitHub. Run 'gh auth login' first."
  exit 1
fi

echo "Searching for workflow runs where job '$JOB_NAME' failed..."
if [[ -n "$WORKFLOW_NAME" ]]; then
  echo "Filtering by workflow: $WORKFLOW_NAME"
fi
if [[ -n "$AFTER_DATE" ]]; then
  echo "Only including runs after: $AFTER_DATE"
fi
echo "Checking last $LIMIT runs..."
echo "----------------------------------------"

# Get workflow runs - only get failed runs to reduce API calls
if [[ -n "$WORKFLOW_NAME" ]]; then
  RUNS=$(gh run list --workflow "$WORKFLOW_NAME" --limit "$LIMIT" --status failure --json databaseId,status,conclusion,displayTitle,createdAt,headBranch,url)
else
  RUNS=$(gh run list --limit "$LIMIT" --status failure --json databaseId,status,conclusion,displayTitle,createdAt,headBranch,url)
fi

# If no failed runs found, exit early
if [[ $(echo "$RUNS" | jq '. | length') -eq 0 ]]; then
  echo "No failed runs found in the last $LIMIT runs."
  exit 0
fi

# Extract run IDs and create a single batch query for all job details
RUN_IDS=()
# Use a temp file instead of associative array for macOS compatibility
RUN_INFO_FILE=$(mktemp)

# First pass: collect run IDs and store run info, applying date filter
while IFS= read -r run; do
  [[ -z "$run" ]] && continue
  
  RUN_ID=$(echo "$run" | jq -r '.databaseId')
  RUN_DATE=$(echo "$run" | jq -r '.createdAt')
  
  # Check date filter if specified
  if [[ -n "$AFTER_DATE" ]]; then
    RUN_EPOCH=$($DATE_CMD -d "$RUN_DATE" +%s 2>/dev/null || echo "0")
    if [[ "$RUN_EPOCH" -lt "$AFTER_EPOCH" ]]; then
      continue
    fi
  fi
  
  RUN_IDS+=("$RUN_ID")
  # Store run info in temp file with RUN_ID as key
  echo "$RUN_ID|$run" >> "$RUN_INFO_FILE"
done < <(echo "$RUNS" | jq -c '.[]')

# If no runs pass the date filter, exit early
if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
  echo "No runs found matching the date criteria."
  exit 0
fi

echo "Found ${#RUN_IDS[@]} failed runs to check for job '$JOB_NAME'..."

# Counter for failed runs found
FAILED_COUNT=0
TEMP_RESULTS=$(mktemp)

# Process runs in smaller batches to avoid command line length limits
BATCH_SIZE=5
for ((i=0; i<${#RUN_IDS[@]}; i+=BATCH_SIZE)); do
  BATCH_IDS=("${RUN_IDS[@]:i:BATCH_SIZE}")
  
  # Process each run in the current batch
  for RUN_ID in "${BATCH_IDS[@]}"; do
    # Get jobs for this specific run (single API call per run, but only for candidates)
    JOBS_OUTPUT=$(gh run view "$RUN_ID" --json jobs 2>/dev/null || echo '{"jobs":[]}')
    
    # Check if the specific job failed using a single jq call
    JOB_FAILED=$(echo "$JOBS_OUTPUT" | jq -r --arg job_name "$JOB_NAME" '
      .jobs[] | 
      select(.name == $job_name and .conclusion == "failure") | 
      .conclusion' | head -1)
    
    if [[ "$JOB_FAILED" == "failure" ]]; then
      # Get run info from temp file
      RUN_DATA=$(grep "^$RUN_ID|" "$RUN_INFO_FILE" | cut -d'|' -f2-)
      
      RUN_TITLE=$(echo "$RUN_DATA" | jq -r '.displayTitle')
      RUN_DATE=$(echo "$RUN_DATA" | jq -r '.createdAt')
      RUN_BRANCH=$(echo "$RUN_DATA" | jq -r '.headBranch')
      RUN_URL=$(echo "$RUN_DATA" | jq -r '.url')
      
      # Format date once
      FORMATTED_DATE=$($DATE_CMD -d "$RUN_DATE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$RUN_DATE")
      
      # Write to temp file for final output
      cat >> "$TEMP_RESULTS" << EOF
❌ FOUND: Run #$RUN_ID failed
   Title: $RUN_TITLE
   Branch: $RUN_BRANCH  
   Date: $FORMATTED_DATE
   Job '$JOB_NAME' status: FAILED
   URL: $RUN_URL

EOF
      
      FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
  done
done

# Output results
if [[ -s "$TEMP_RESULTS" ]]; then
  cat "$TEMP_RESULTS"
else
  echo "No runs found where job '$JOB_NAME' failed."
fi

echo "=== SUMMARY REPORT ==="
echo "Total runs with job '$JOB_NAME' failures: $FAILED_COUNT"

# Cleanup
rm -f "$TEMP_RESULTS" "$RUN_INFO_FILE"