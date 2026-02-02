#date: 2026-02-02T17:16:36Z
#url: https://api.github.com/gists/a231292d26e5a995823b043227af101a
#owner: https://api.github.com/users/ajponte

#!/bin/bash

SESSION_NAME="******"
PROJECT_DIR="$HOME/******"

# Check if the session already exists, if not create it
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    # Create a new detached session in the project directory
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"

    # Rename the initial window
    tmux rename-window -t "$SESSION_NAME:1" 'main'
    # Run a command in the main window (e.g., list files)
    # tmux send-keys -t "$SESSION_NAME:main" 'ls -lh' C-m

    # Create a new window named 'dev'
    # tmux new-window -t "$SESSION_NAME:1" -n 'dev' -c "$PROJECT_DIR/src"
    # Split the new window horizontally
    # tmux split-window -h -t "$SESSION_NAME:dev"
    # Run a command in the top pane (e.g., open vim)
    # tmux send-keys -t "$SESSION_NAME:dev.0" 'vim main.c' C-m
    # Run a command in the bottom pane (e.g., run a server)
    # tmux send-keys -t "$SESSION_NAME:dev.1" './start_server.sh' C-m
fi

# Attach to the session when the script is run
tmux attach-session -t "$SESSION_NAME"

