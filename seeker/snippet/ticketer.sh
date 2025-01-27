#date: 2025-01-27T16:58:04Z
#url: https://api.github.com/gists/58e0ca95dd25da44fedf52a857e8ad85
#owner: https://api.github.com/users/RoseSecurity

#!/usr/bin/env bash

# Generate Jira tickets programmatically
# Requires gum and jira-cli for interactivity
# Install gum: brew install gum

if ! command -v gum &>/dev/null; then
  echo "Error: Gum is not installed. Install it with 'brew install gum'."
  exit 1
fi

# Variables
TYPE="Task"

# Get priority
PRIORITY=$(gum choose --limit=1 --header="Select Priority" "Lowest" "Low" "Medium" "High" "Highest")

# Get title
TITLE=$(gum input --placeholder "Enter Ticket Title")
if [ -z "$TITLE" ]; then
  echo "Error: Ticket Title cannot be empty."
  exit 1
fi

# Get labels
LABELS=$(gum input --placeholder "Enter labels for this ticket (optional)")

# Get description
DESCRIPTION=$(gum write --placeholder "Enter details of this ticket (optional)")

# Confirm and create the ticket
if gum confirm "Do you want to create this ticket?"; then
  jira issue create \
    -t "$TYPE" \
    -s "$TITLE" \
    -y "$PRIORITY" \
    -b "$DESCRIPTION" \
    -l "$LABELS" \
    -a "$(jira me)" \
    --no-input
  if [ $? -eq 0 ]; then
    echo "Ticket created successfully!"
  else
    echo "Error: Failed to create the ticket."
    exit 1
  fi
else
  echo "Ticket creation cancelled."
fi