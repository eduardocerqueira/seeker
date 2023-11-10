#date: 2023-11-10T16:36:00Z
#url: https://api.github.com/gists/515eb7ec704968e691a721e67cfa65f8
#owner: https://api.github.com/users/sebastiancarlos

#!/usr/bin/env bash

# All my gist code is licensed under the MIT license.

# Add this to your PATH

# print usage with -h/--help
if [[ "$#" -eq 1 && ($1 == "-h" || $1 == "--help") ]]; then
    echo "Usage: sway-get-workspace-cwd [workspace-name]"
    echo "  - Prints the CWD of the first window in the provided workspace"
    echo "  - If no workspace is provided, uses the currently focused workspace"
    echo "  - If there is no window in the workspace, return ~"
    exit 1
fi

# if workspace name was provided, use it. else, get focused workspace name
if [[ -n $1 ]]; then
    workspace=$1
else
    workspace=$(swaymsg -t get_workspaces | jq '.[] | select(.focused==true).name' | tr -d '"')
fi

# get first window in workspace 
# - By getting every node in the workspace (even nested nodes), and
#   returning the first one with a PID. 
# - This assumes that the first window is a terminal.
first_window_pid=$(swaymsg -t get_tree | jq -r --arg workspace "$workspace" '.nodes[].nodes[] | select(.name==$workspace) | recurse(.nodes[]) | del(.nodes) | select(.pid) | .pid' | head -n 1)

# if pid was found, get its cwd. else return $HOME
if [[ -n $first_window_pid ]]; then
    echo $(get-terminal-cwd $first_window_pid)
else
    echo "$HOME"
fi

