#date: 2025-01-13T16:52:41Z
#url: https://api.github.com/gists/0677f6cc12c94d799356d4d2ddf7a4b5
#owner: https://api.github.com/users/labmonkey

#!/bin/sh

# Note: In AsusWRT you cannot use bash but just sh

discord_url="https://discord.com/api/webhooks/your-webhook"

# Manually set variables for testing
# script_type="client-connect"
# common_name="someuser"

# space seperated list as string
# In case you have some client that you don't want to see notifications of then you can disable it for this user here
# blacklist="item1 item2 item3"
blacklist="baduser"

generate_post_data() {
    for item in $blacklist; do
      if [ "$item" = "$common_name" ]; then
        echo "User '$common_name' is in the blacklist. Skipping notification." >&2
        return 0
      fi
    done
    
    if [ "$script_type" = "client-connect" ]; then
        echo "{\"content\": \"User $common_name connected with IP $trusted_ip\"}"
    elif [ "$script_type" = "client-disconnect" ]; then
        echo "{\"content\": \"User $common_name disconnected with IP $trusted_ip\"}"
    fi
}

# Generate JSON data
data=$(generate_post_data)

if [ -z "$data" ]; then
    echo "No data to send. Exiting." >&2
    exit 1
fi

# Send notification to Discord Webhook
echo "Sending notification to Discord..."
response=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" -X POST -d "$data" "$discord_url")

if [ "$response" -eq 204 ]; then
    echo "Notification sent successfully."
else
    echo "Failed to send notification. HTTP status code: $response" >&2
fi