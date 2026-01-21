#date: 2026-01-21T17:51:24Z
#url: https://api.github.com/gists/7246b5b99db4108b2e74d11488522390
#owner: https://api.github.com/users/chrismerck

#!/bin/bash

# Read JSON input from stdin
input=$(cat)

# Extract model information
model=$(echo "$input" | jq -r '.model.display_name // "Unknown"')

# Extract context usage percentage
used_pct=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

# Extract turn count
turn_count=$(echo "$input" | jq -r '.turn_count // empty')

# Extract session duration (seconds)
session_secs=$(echo "$input" | jq -r '.session_duration_seconds // empty')

# Calculate approximate session cost based on token usage and model
total_input= "**********"
total_output= "**********"

# Approximate pricing (as of Jan 2025, per million tokens)
cost=0
model_id=$(echo "$input" | jq -r '.model.id // ""')

if [[ $model_id == *"opus"* ]]; then
    # Opus: "**********"
    cost=$(awk "BEGIN {printf \"%.3f\", ($total_input * 0.000015) + ($total_output * 0.000075)}")
elif [[ $model_id == *"sonnet"* ]]; then
    # Sonnet: "**********"
    cost=$(awk "BEGIN {printf \"%.3f\", ($total_input * 0.000003) + ($total_output * 0.000015)}")
elif [[ $model_id == *"haiku"* ]]; then
    # Haiku: "**********"
    cost=$(awk "BEGIN {printf \"%.4f\", ($total_input * 0.00000025) + ($total_output * 0.00000125)}")
fi

# Colors
YELLOW=$'\033[0;33m'
GREEN=$'\033[0;32m'
CYAN=$'\033[0;36m'
DARKGREY=$'\033[0;90m'
GREY=$'\033[38;5;245m'
GOLD=$'\033[38;5;136m'
DARKBLUE=$'\033[0;34m'
MAGENTA=$'\033[0;35m'
RESET=$'\033[0m'

# Git branch
git_branch=$(git branch --show-current 2>/dev/null)
git_part=""
if [[ -n "$git_branch" ]]; then
    git_part=" ${MAGENTA}{${git_branch}}${RESET}"
fi

# Format session duration as HH:MM:SS or MM:SS
duration_part=""
if [[ -n "$session_secs" && "$session_secs" != "null" ]]; then
    secs=$((session_secs % 60))
    mins=$(((session_secs / 60) % 60))
    hours=$((session_secs / 3600))
    if [[ $hours -gt 0 ]]; then
        duration_part=" ${GREY}${hours}h${mins}m${RESET}"
    elif [[ $mins -gt 0 ]]; then
        duration_part=" ${GREY}${mins}m${secs}s${RESET}"
    else
        duration_part=" ${GREY}${secs}s${RESET}"
    fi
fi

# Turn count
turn_part=""
if [[ -n "$turn_count" && "$turn_count" != "null" ]]; then
    turn_part=" ${GREY}#${turn_count}${RESET}"
fi

# Build model/context/cost part: dark grey except gold for percentage
model_part=""
if [[ -n "$used_pct" ]]; then
    model_part="${DARKGREY}[${DARKBLUE}${model} ${GOLD}${used_pct}%${DARKGREY}]${RESET}"
else
    model_part="${DARKGREY}[${DARKBLUE}${model}${DARKGREY}]${RESET}"
fi

if [[ $(echo "$cost > 0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
    model_part="$model_part ${GREY}\$${cost}${RESET}"
fi

# Output: user@host:path branch  [model pct%] $cost #turns duration
printf "%s%s%s@%s%s%s:%s%s%s%s %s%s%s" \
    "$YELLOW" "$(whoami)" "$RESET" \
    "$GREEN" "$(hostname -s)" "$RESET" \
    "$CYAN" "$(pwd)" "$RESET" \
    "$git_part" \
    "$model_part" \
    "$turn_part" \
    "$duration_part"
%s%s %s%s%s" \
    "$YELLOW" "$(whoami)" "$RESET" \
    "$GREEN" "$(hostname -s)" "$RESET" \
    "$CYAN" "$(pwd)" "$RESET" \
    "$git_part" \
    "$model_part" \
    "$turn_part" \
    "$duration_part"
