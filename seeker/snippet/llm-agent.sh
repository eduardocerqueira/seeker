#date: 2024-03-08T17:03:17Z
#url: https://api.github.com/gists/a242b07e011e2c1d003a19c9a735a3b2
#owner: https://api.github.com/users/CJHwong

#!/bin/zsh

# Path to the LLM executable (replace with actual path if different)
LLM_COMMAND="/opt/homebrew/bin/llm"

# Name of the LLM model to use (change to desired model name)
LLM_MODEL="mistral"

# Define available agents and their prompts
declare -A agentMap
agentMap["Passthrough"]=" "
agentMap["EditingAssistant"]="Act as an English editing assistant. I'll provide a piece of text, and your job is to suggest improvements to its grammar, word choice, and sentence structure. Aim for clear communication with a neutral tone and a natural conversational style. For each suggestion, explain what you changed and why it would improve the text. Text:"
agentMap["ExplainCode"]="Break down the following code line by line. Explain what each line does and how it contributes to the overall logic of the code. If possible, describe any potential business applications of this code. Try reasoning the code step by step. Code:"
## Add more agents here
# agentMap["AgentName"]="Your Prompt"

# Build list of agents from the map (to display in the AppleScript prompt)
agents=""
for key in ${(k)agentMap}
do
    if [[ $agents == "" ]]; then
        agents="$key"
        continue
    fi
    agents="$agents, $key"
done

# Prompt user to choose an agent using osascript
selected_agent=$(osascript -e "choose from list {$agents} with prompt \"Select an agent to use:\" default items {\"Passthrough\"}")

# Build the prompt based on chosen agent and escaped arguments
escaped_args=""
for arg in "$@"; do
    escaped_arg=$(printf '%s\n' "$arg" | sed "s/'/'\\\\''/g")
    escaped_args="$escaped_args '$escaped_arg'"
done
prompt="${agentMap[\"$selected_agent\"]} $escaped_args"

# Run LLM command, escape result for safety
result=$($LLM_COMMAND -m $LLM_MODEL $prompt)
escapedResult=$(echo "$result" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | awk '{printf "%s\\n", $0}' ORS='')

# Copy result to clipboard and display notification
osascript -e "set the clipboard to \"$escapedResult\""
osascript -e "display notification \"LLM results copied to clipboard\""

# Open iTerm window with temporary file for displaying result
temp_file=$(mktemp)
echo "$escapedResult" > "$temp_file"

osascriptCmd="tell application \"iTerm2\"
    create window with default profile
    tell current session of current window
        write text \"glow -p ${temp_file} && exit\"
    end tell
end tell"
osascript -e "$osascriptCmd"
