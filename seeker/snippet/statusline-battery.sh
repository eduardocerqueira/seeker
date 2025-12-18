#date: 2025-12-18T17:07:54Z
#url: https://api.github.com/gists/e3904bbb9c1b5617da5c1c870aa784f1
#owner: https://api.github.com/users/szerintedmi

#!/bin/bash
#
# Claude Code Custom Statusline
# ==============================
# A battery-style context indicator with git status and lines of code.
#
# Features:
#   - Context usage bar matching /context command style
#     - Purple ⛁ = "**********"
#     - Gray ⛶ = free space
#     - Gray ⛝ = autocompact buffer (22.5%)
#   - Free space % (of usable context, excluding autocompact buffer)
#   - Git status: [branch ↑ahead↓behind ●staged ✖conflicts ✚changed ⚑stashed]
#   - Lines of code: green +added, red -removed
#
# Example output:
#   Claude Opus 4.5 ⛁⛁⛁⛁⛁⛶⛶⛶⛶⛶⛶⛝⛝⛝ 67% │ ~/project [main ↑2 ●3] │ +127 -45
#
# Installation:
#   1. Download this script:
#      curl -o ~/.claude/statusline-battery.sh https://gist.githubusercontent.com/szerintedmi/e3904bbb9c1b5617da5c1c870aa784f1/raw/statusline-battery.sh
#
#   2. Make it executable:
#      chmod +x ~/.claude/statusline-battery.sh
#
#   3. Add to ~/.claude/settings.json:
#      {
#        "statusLine": {
#          "type": "command",
#          "command": "bash ~/.claude/statusline-battery.sh"
#        }
#      }
#
# Requirements: jq, git
#

# Read the JSON input from stdin
input=$(cat)

# Extract values
model=$(echo "$input" | jq -r '.model.display_name')
cwd=$(echo "$input" | jq -r '.workspace.current_dir')
usage=$(echo "$input" | jq '.context_window.current_usage')

# Abbreviate home directory with ~
display_path="$cwd"
if [[ "$cwd" == "$HOME"* ]]; then
    display_path="~${cwd#$HOME}"
fi

# Get git super status style info with colors
git_info=""
if git -C "$cwd" rev-parse --git-dir > /dev/null 2>&1; then
    branch=$(git -C "$cwd" --no-optional-locks rev-parse --abbrev-ref HEAD 2>/dev/null)

    if [ -n "$branch" ]; then
        # Get git status indicators
        ahead=$(git -C "$cwd" --no-optional-locks rev-list @{u}..HEAD 2>/dev/null | wc -l | tr -d ' ')
        behind=$(git -C "$cwd" --no-optional-locks rev-list HEAD..@{u} 2>/dev/null | wc -l | tr -d ' ')
        staged=$(git -C "$cwd" --no-optional-locks diff --cached --numstat 2>/dev/null | wc -l | tr -d ' ')
        conflicts=$(git -C "$cwd" --no-optional-locks diff --name-only --diff-filter=U 2>/dev/null | wc -l | tr -d ' ')
        changed=$(git -C "$cwd" --no-optional-locks diff --numstat 2>/dev/null | wc -l | tr -d ' ')
        untracked=$(git -C "$cwd" --no-optional-locks ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')
        stashed=$(git -C "$cwd" --no-optional-locks stash list 2>/dev/null | wc -l | tr -d ' ')

        # Build git status string with symbols and colors
        # Colors: cyan for branch, yellow for ahead/behind, green for staged, red for conflicts, blue for changed, magenta for stashed
        git_status=""
        [ "$ahead" != "0" ] && git_status="${git_status}$(printf '\033[33m')↑${ahead}$(printf '\033[0m')"
        [ "$behind" != "0" ] && git_status="${git_status}$(printf '\033[33m')↓${behind}$(printf '\033[0m')"
        [ "$staged" != "0" ] && git_status="${git_status}$(printf '\033[32m')●${staged}$(printf '\033[0m')"
        [ "$conflicts" != "0" ] && git_status="${git_status}$(printf '\033[31m')✖${conflicts}$(printf '\033[0m')"
        [ "$changed" != "0" ] && git_status="${git_status}$(printf '\033[34m')✚${changed}$(printf '\033[0m')"
        [ "$stashed" != "0" ] && git_status="${git_status}$(printf '\033[35m')⚑${stashed}$(printf '\033[0m')"

        # Format: [branch status] with cyan branch name
        if [ -n "$git_status" ]; then
            git_info=" [$(printf '\033[36m')${branch}$(printf '\033[0m') ${git_status}]"
        else
            git_info=" [$(printf '\033[36m')${branch}$(printf '\033[0m')]"
        fi
    fi
fi

# Build the battery-style context indicator
# Note: Detailed breakdown (system prompt, tools, MCP, agents, memory, messages)
# is not available in statusline JSON - only aggregate token counts
context_bar=""
if [ "$usage" != "null" ]; then
    # Get token counts
    input_tokens= "**********"
    cache_creation= "**********"
    cache_read= "**********"

    # Total tokens used
    total_used= "**********"
    context_window_size=$(echo "$input" | jq '.context_window.context_window_size')

    # Autocompact buffer is ~22.5% of context window (constant)
    autocompact_pct=225  # 22.5% * 10 for integer math
    autocompact_buffer=$((context_window_size * autocompact_pct / 1000))

    # Usable space = total - autocompact buffer
    usable_space=$((context_window_size - autocompact_buffer))

    # Actual free space (what you can still use)
    actual_free=$((usable_space - total_used))
    if [ $actual_free -lt 0 ]; then actual_free=0; fi

    # Calculate actual free percentage of usable space
    actual_free_pct=$((actual_free * 100 / usable_space))

    # Battery bar width (16 characters for better resolution)
    bar_width=16

    # Calculate portions (of total context window for accurate visual)
    used_portion=$((total_used * bar_width / context_window_size))
    autocompact_portion=$((autocompact_buffer * bar_width / context_window_size))
    free_portion=$((bar_width - used_portion - autocompact_portion))
    if [ $free_portion -lt 0 ]; then free_portion=0; fi

    # Build colored bar matching /context colors
    colored_bar=""

    # Purple section (used tokens) - RGB 147,51,234
    if [ $used_portion -gt 0 ]; then
        colored_bar="${colored_bar}$(printf '\033[38;2;147;51;234m')"
        for ((i=0; i<used_portion; i++)); do
            colored_bar="${colored_bar}⛁"
        done
        colored_bar="${colored_bar}$(printf '\033[0m')"
    fi

    # Gray hollow section (actual free space) - ⛶
    if [ $free_portion -gt 0 ]; then
        colored_bar="${colored_bar}$(printf '\033[38;2;153;153;153m')"
        for ((i=0; i<free_portion; i++)); do
            colored_bar="${colored_bar}⛶"
        done
        colored_bar="${colored_bar}$(printf '\033[0m')"
    fi

    # Gray crossed section (autocompact buffer) - ⛝ matching /context
    if [ $autocompact_portion -gt 0 ]; then
        colored_bar="${colored_bar}$(printf '\033[38;2;153;153;153m')"
        for ((i=0; i<autocompact_portion; i++)); do
            colored_bar="${colored_bar}⛝"
        done
        colored_bar="${colored_bar}$(printf '\033[0m')"
    fi

    context_bar=" ${colored_bar} ${actual_free_pct}%"
fi

# Get lines of code stats
lines_added=$(echo "$input" | jq '.cost.lines_of_code.added // 0')
lines_removed=$(echo "$input" | jq '.cost.lines_of_code.removed // 0')

# Build lines of code indicator (green +x, red -x)
loc_info=""
if [ "$lines_added" != "0" ] || [ "$lines_removed" != "0" ]; then
    loc_info=" $(printf '\033[38;2;153;153;153m')│$(printf '\033[0m') "
    [ "$lines_added" != "0" ] && loc_info="${loc_info}$(printf '\033[32m')+${lines_added}$(printf '\033[0m')"
    [ "$lines_added" != "0" ] && [ "$lines_removed" != "0" ] && loc_info="${loc_info} "
    [ "$lines_removed" != "0" ] && loc_info="${loc_info}$(printf '\033[31m')-${lines_removed}$(printf '\033[0m')"
fi

# Separator character (box drawing light vertical)
sep="$(printf '\033[38;2;153;153;153m')│$(printf '\033[0m')"

# Output the status line
# Format: model [context_bar] │ path git_info │ +added -removed
printf "%s%s %s %s%s%s\n" "$model" "$context_bar" "$sep" "$display_path" "$git_info" "$loc_info"
ntext_bar] │ path git_info │ +added -removed
printf "%s%s %s %s%s%s\n" "$model" "$context_bar" "$sep" "$display_path" "$git_info" "$loc_info"
