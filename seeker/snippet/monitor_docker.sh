#date: 2025-04-07T16:47:27Z
#url: https://api.github.com/gists/48263147f9b7e489e8f2a7d11ca8f286
#owner: https://api.github.com/users/alisafariir

# create /usr/local/bin/monitor_docker.sh
# sudo chmod +x /usr/local/bin/monitor_docker.sh

#!/bin/bash

# Telegram bot settings
BOT_TOKEN= "**********"
CHAT_ID="your_chat_id_here"

# Function to send Telegram message
send_notification() {
    local event_type=$1
    local container_name=$2
    local additional_info=$3
    
    case $event_type in
        "start")
            local emoji="🟢"
            local action="Started"
            ;;
        "stop")
            local emoji="🔴"
            local action="Stopped"
            ;;
        "die")
            local emoji="⚫"
            local action="Crashed"
            ;;
        "restart")
            local emoji="🟡"
            local action="Restarted"
            ;;
        *)
            local emoji="ℹ️"
            local action="Event"
            ;;
    esac

    local message="${emoji} *Docker Container ${action}* ${emoji}
*Container:* ${container_name}
*Host:* $(hostname)
*Time:* $(date)"

    # Add additional info if provided
    if [ -n "$additional_info" ]; then
        message="${message}
${additional_info}"
    fi

    curl -s -X POST "https: "**********"
        -d chat_id="${CHAT_ID}" \
        -d text="${message}" \
        -d parse_mode="Markdown"
}

# Monitor Docker events
docker events --format '{{json .}}' | while read -r event; do
    status=$(echo "$event" | jq -r '.status')
    container_name=$(echo "$event" | jq -r '.Actor.Attributes.name')
    
    case $status in
        "start")
            send_notification "start" "$container_name"
            ;;
        "stop")
            send_notification "stop" "$container_name"
            ;;
        "die")
            exit_code=$(echo "$event" | jq -r '.Actor.Attributes.exitCode')
            send_notification "die" "$container_name" "*Exit Code:* ${exit_code}"
            ;;
        "restart")
            send_notification "restart" "$container_name"
            ;;
        # Add more event types if needed
    esac
doneAdd more event types if needed
    esac
done