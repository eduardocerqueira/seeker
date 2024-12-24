#date: 2024-12-24T17:11:28Z
#url: https://api.github.com/gists/111166383c2abc5e216429e6fd5103c1
#owner: https://api.github.com/users/NohamR

#!/bin/bash

# Function to convert bytes to human-readable format
convert_bytes() {
    local bytes=$1

    # Convert to megabytes
    megabytes=$(echo "scale=3; $bytes / 1024 / 1024" | bc)

    # Convert to gigabytes
    gigabytes=$(echo "scale=3; $megabytes / 1024" | bc)

    echo "$megabytes Mo" #($gigabytes GB)
}

# Function to send Discord notification
send_discord_notification() {
    local webhook_url=$1
    local torrent_name=$2
    local save_path=$3
    local torrent_size=$4

    size_readable=$(convert_bytes "$torrent_size")
    payload="{
        \"embeds\": [
            {
                \"title\": \"New Torrent Downloaded\",
                \"fields\": [
                    { \"name\": \"Name\", \"value\": \"$torrent_name\", \"inline\": true },
                    { \"name\": \"Save Path\", \"value\": \"$save_path\", \"inline\": true },
                    { \"name\": \"Size\", \"value\": \"$size_readable\", \"inline\": true }
                ]
            }
        ]
    }"
    curl -sS -X POST -H "Content-Type: application/json" -d "$payload" "$webhook_url"
}

# Check if all arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <Torrent Name> <Save Path> <Torrent Size>"
    exit 1
fi

# Parse arguments
torrent_name=$1
save_path=$2
torrent_size=$3

# Replace any double quotes in the arguments
torrent_name=${torrent_name//\"/\\\"}
save_path=${save_path//\"/\\\"}

# Discord Webhook URL
webhook_url=$DISCORD_WEBHOOK_URL

# Call the function to send Discord notification
send_discord_notification "$webhook_url" "$torrent_name" "$save_path" "$torrent_size"