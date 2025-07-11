#date: 2025-07-11T16:55:22Z
#url: https://api.github.com/gists/10e73d4d744a0775f4c01cafdb4852ec
#owner: https://api.github.com/users/CJHwong

#!/bin/bash

# Usage: ./gpt-4.1.sh "<user query>"
# Sends query to opencode with gpt-4.1 and extracts clean output in real-time

if [ -z "$1" ]; then
  echo "Usage: $0 \"<user query>\""
  exit 1
fi

USER_QUERY="$1"
MODEL="github-copilot/gpt-4.1"

# Enhanced query with output wrapping instruction
WRAPPED_QUERY="$USER_QUERY

IMPORTANT: Wrap your entire response in <opencode_output></opencode_output> tags. Put ALL your output content inside these tags."

# Create a temporary file to capture all output for fallback
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

CMD="opencode --model '$MODEL' run '$WRAPPED_QUERY'"

printf "\r" # Send a carriage return to ensure terminal is ready

# Capture both stdout and stderr to see what's happening
eval $CMD 2>&1 | tee "$TEMP_FILE" | {
  inside=false
  after_model_line=false
  found_start=false
  first_output_line=false

  while IFS= read -r line; do
    # Strip ANSI escape codes
    clean_line=$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g')

    # Check for start tag (more flexible matching)
    if [[ "$clean_line" == *"$MODEL"* ]]; then
      after_model_line=true
      continue
    fi

    if [[ "$clean_line" == *"<opencode_output>"* ]]; then
      inside=true
      found_start=true
      continue
    fi

    # Check for end tag
    if [[ "$clean_line" == *"</opencode_output>"* ]]; then
      if [ "$inside" = true ]; then
        break
      fi
    fi

    # Output lines when inside tags
    if [ "$inside" = true ]; then
      # Skip empty lines only before the first non-empty line
      if [[ -z "$clean_line" && "$first_output_line" != true ]]; then
        continue
      fi
      if [[ -n "$clean_line" ]]; then
        first_output_line=true
      fi
      printf "%s\n" "$clean_line"
    fi
  done

  # Fallback if no tags found - show everything we captured
  if [ "$found_start" = false ]; then
    echo "No tags found, showing all captured output:" >&2
    cat "$TEMP_FILE"
  fi
}
