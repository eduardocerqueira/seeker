#date: 2025-04-25T16:53:57Z
#url: https://api.github.com/gists/db88bdae676a33b97eaf09507dc88780
#owner: https://api.github.com/users/bgaeddert

#!/bin/bash

# Default values if not provided
SERVER_URL=${1:-"https://local-cronicle.schoolcurrent.net:8443"}
API_KEY=${2:-"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

# Display which server and API key we're using
echo "Using Cronicle server: $SERVER_URL"
echo "Using API Key: $API_KEY"

# Get event details and extract ID
EVENT_ID=$(curl -s -X GET "$SERVER_URL/api/app/get_event/v1?title=RebootDMS" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" | jq -r '.event.id')

echo "Found event ID: $EVENT_ID"

# Check if we got a valid ID
if [ -z "$EVENT_ID" ] || [ "$EVENT_ID" = "null" ]; then
  echo "Error: Could not retrieve event ID"
  exit 1
fi

# Update the event (disable)
curl -X POST "$SERVER_URL/api/app/update_event/v1" \
  -d "{\"id\": \"$EVENT_ID\", \"enabled\": 0}" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json"
