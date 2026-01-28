#date: 2026-01-28T17:29:14Z
#url: https://api.github.com/gists/d18457200bdc400975e277e8350a7df5
#owner: https://api.github.com/users/ishraq10199

#!/usr/bin/env bash

# -------------------------------------------------------------------
# 
# Single window, three panes a, b, c
# 
# a => Dev Server
# b => Object storage server (Garage S3)
# c => Ngrok session as a webhook listener (e.g for stripe/paddle)
# 
# ┌─────┬─────┐
# │     │  b  │
# │  a  ├─────┤
# │     │  c  │
# └─────┴─────┘
# 
# Note: `NGROK_URL` should be defined in the .env file
#        It should contain the free-tier URL as its value
# 
# -------------------------------------------------------------------

# Hardcoded, defined in the .env file, or just passed in as an arg
session="ADD_A_SESSION_NAME"

# Default behavior: If the session does not exist, then initialize

if ! tmux has-session -t "$session" 2>/dev/null; then

# To use .env file variables
source .env

# Session init
tmux new -Ads "$session"
tmux rename-window -t "$session":0 'DevConsole'

# Pane a
tmux send-keys -t "$session":0 'composer run dev' C-m
tmux split-window -h -t "$session":0

# Pane b
tmux select-pane -t "$session":0.1 -T 'S3'
tmux send-keys -t "$session":0.1 'garage server' C-m
tmux split-window -v -t "$session":0.1

# Pane c
tmux select-pane -t "$session":0.2 -T 'Webhooks'
tmux send-keys -t "$session":0.2 "ngrok http 8000 --url \"$NGROK_URL\"" C-m

# Session options
tmux set-option -t "$session":0 mouse on

# Select the first pane
tmux select-pane -t "$session":0.0 -T 'DevConsole'

fi

# After initialization (or if already initialized), attach

tmux a -t $session