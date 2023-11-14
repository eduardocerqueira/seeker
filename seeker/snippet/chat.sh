#date: 2023-11-14T17:06:07Z
#url: https://api.github.com/gists/46a3790339e5bd7a8f59572f298ccdb6
#owner: https://api.github.com/users/cwchriswilliams

#!/bin/bash

# Function to check if required commands are installed
check_dependencies() {
  for cmd in curl jq; do
    if ! command -v "$cmd" &> /dev/null; then
      echo "Error: $cmd is not installed." >&2
      exit 1
    fi
  done
}

# Function to load system message from a file
load_system_message() {
  if [[ ! -f "system_message.txt" ]]; then
    echo "Error: System message file not found." >&2
    exit 1
  fi
  jq -aRs . "system_message.txt"
}

# Function to escape JSON strings
escape_json() {
  echo "$1" | jq -aRs .
}

# Function to send message to OpenAI API and get the response
send_message() {
  local input="$1"
  local escaped_input=$(escape_json "$input")

  # Append message to the message sequence
  messages="${messages}, {\"role\": \"user\", \"content\": $escaped_input}"

  # Send to API
  local response=$(curl -s "https://api.openai.com/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -d "{
        \"model\": \"gpt-3.5-turbo\",
        \"max_tokens\": "**********"
        \"messages\": [$messages]
      }")

  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to connect to the OpenAI API." >&2
    exit 1
  fi

  # Extract the content from the API response
  local content=$(echo "$response" | jq -r '.choices[0].message.content')
  echo "$content"

  local escaped_content=$(escape_json "$content")
  messages="${messages}, {\"role\": \"assistant\", \"content\": $escaped_content}"
}

# Main script starts here
check_dependencies

# Check for OPENAI_API_KEY environment variable
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Please set the OPENAI_API_KEY environment variable." >&2
  exit 1
fi

# Load system message and initialize message sequence
system_message=$(load_system_message)
messages="{\"role\": \"system\", \"content\": $system_message}"

chat_mode=false
# Check for chat mode
if [ "$1" == "--chat" ]; then
  chat_mode=true
  shift # Remove '--chat' from arguments
fi

# Get initial input from remaining arguments or prompt the user
if [ "$#" -gt 0 ]; then
  user_input="$*"
else
  read -rp "Input: " user_input
fi

# If chat mode is on, loop indefinitely
if $chat_mode; then
  counter=0
  while true; do
    if [ -z "$user_input" ]; then
      echo "Empty input detected. Please enter a valid message."
      continue
    fi

    send_message "$user_input"

    ((counter++))
    if ((counter >= 10)); then
      echo "Maximum conversation limit reached. Exiting..."
      break
    fi

    printf "\nWould you like to know more? (Type 'exit' to quit)\n"
    read -r user_input

    # Exit condition
    if [[ "$user_input" == "exit" || "$user_input" == "quit" ]]; then
      echo "Exiting..."
      break
    fi
  done
else
  # If not in chat mode, send a single message
  send_message "$user_input"
fi
