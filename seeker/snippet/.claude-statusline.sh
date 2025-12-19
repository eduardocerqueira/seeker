#date: 2025-12-19T17:11:40Z
#url: https://api.github.com/gists/48c93345041dd2ed73fc0b3bceebda27
#owner: https://api.github.com/users/nlaffey

#!/bin/bash
# Claude Code status line script - Apple theme style
# Receives JSON on stdin, outputs status line matching zsh apple theme
# Shows: directory [branch*+] (session-id)

# Read JSON from stdin
json=$(cat)

# Parse relevant fields using jq
if command -v jq &>/dev/null && [ -n "$json" ]; then
    cwd=$(echo "$json" | jq -r '.workspace.current_dir // .cwd // empty' 2>/dev/null)

    # Extract session ID - try multiple possible field locations
    session_id=$(echo "$json" | jq -r '.session_id // empty' 2>/dev/null)

    # Fallback: try extracting from transcript_path
    if [ -z "$session_id" ]; then
        transcript_path=$(echo "$json" | jq -r '.transcript_path // .session.transcript_location // empty' 2>/dev/null)
        if [ -n "$transcript_path" ]; then
            session_dir=$(dirname "$transcript_path")
            session_id=$(basename "$session_dir")
        fi
    fi

    # Keep full session ID for `claude --continue` usage
    # (no truncation)

    # Replace $HOME with ~ for display (no backslash escape)
    if [ -n "$cwd" ]; then
        display_dir="${cwd/#$HOME/~}"
    else
        display_dir="~"
    fi

    # Get git branch info if in a git repository
    git_info=""
    if [ -n "$cwd" ] && [ -d "$cwd/.git" ] || git -C "$cwd" rev-parse --git-dir > /dev/null 2>&1; then
        branch=$(git -C "$cwd" symbolic-ref --short HEAD 2>/dev/null || git -C "$cwd" rev-parse --short HEAD 2>/dev/null)

        if [ -n "$branch" ]; then
            # Check for unstaged changes (--no-optional-locks prevents index.lock conflicts)
            unstaged=""
            if ! git -C "$cwd" --no-optional-locks diff --quiet 2>/dev/null; then
                unstaged="*"
            fi

            # Check for staged changes (--no-optional-locks prevents index.lock conflicts)
            staged=""
            if ! git -C "$cwd" --no-optional-locks diff --cached --quiet 2>/dev/null; then
                staged="+"
            fi

            # Format git info with colors (using printf for ANSI codes)
            # Colors: dim magenta for brackets, dim green for branch, dim red for unstaged, dim yellow for staged
            git_info=$(printf " \033[2;35m[\033[2;32m%s\033[2;31m%s\033[2;33m%s\033[2;35m]\033[0m" "$branch" "$unstaged" "$staged")
        fi
    fi

    # Build session info if available (dim cyan color)
    session_info=""
    if [ -n "$session_id" ]; then
        session_info=$(printf " \033[2;36m(%s)\033[0m" "$session_id")
    fi

    # Read topic from ~/.claude-topic if it exists (dim white/gray color)
    topic_info=""
    if [ -f "$HOME/.claude-topic" ]; then
        topic=$(cat "$HOME/.claude-topic" 2>/dev/null | head -1 | cut -c1-30)
        if [ -n "$topic" ]; then
            topic_info=$(printf " \033[2;37m│ %s\033[0m" "$topic")
        fi
    fi

    # Build status line: directory + git info + session id + topic
    printf "\033[2;35m%s\033[0m%s%s%s" "$display_dir" "$git_info" "$session_info" "$topic_info"
else
    # No jq or no json, show simple prompt
    echo "~/"
fi
