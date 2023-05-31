#date: 2023-05-31T16:57:51Z
#url: https://api.github.com/gists/1621650728c32723dc6d02d98da0c105
#owner: https://api.github.com/users/green-leader

#!/bin/bash

readonly WEBHOOK_URL="https://discord.com/api/webhooks/0000/XXXX"

# List of directories to check
readonly WATCH_DIRS=(
  "/mnt/dir1"
  "/mnt/dir2"
  "/mnt/dir3"
)

# Function to send a Discord webhook message
send_discord_message() {
  local message="$1"
  local payload="{\"content\": \"$message\"}"

  curl -H "Content-Type: application/json" -d "$payload" "$WEBHOOK_URL"
}

# Iterate over the directories
for dir in "${WATCH_DIRS[@]}"; do
  # Check if any files were modified in the last 7 days
  if find "$dir" -type f -newermt "-7 days" 2>/dev/null | grep -q .; then
    continue
  fi

  # Send Discord webhook message for directories without recent changes
  
  MESSAGE="Directory $dir has not been modified in the last 7 days"
  send_discord_message "$MESSAGE"
  echo "$MESSAGE"
done
