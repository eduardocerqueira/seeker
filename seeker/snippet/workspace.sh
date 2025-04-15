#date: 2025-04-15T16:31:12Z
#url: https://api.github.com/gists/e7c4520ecfbbbb250bf30efd120447fd
#owner: https://api.github.com/users/volker-fr

#!/bin/bash

# Usage:
# alt-0 = 'exec-and-forget bash ~/repos/dotfiles/bin/workspace.sh 10'
# alt-1 = 'exec-and-forget bash ~/repos/dotfiles/bin/workspace.sh 1'
# alt-2 = 'exec-and-forget bash ~/repos/dotfiles/bin/workspace.sh 2'
# ...


set -e
set -u
set -o pipefail

# log everything to /tmp/workspace.log including stderr
exec > >(tee -a /tmp/workspace.log) 2>&1

WORKSPACE="$1"

# if value of $(aerospace. ..) is not 0, then do nothing
WINDOW_COUNT="$(aerospace list-windows --workspace "$WORKSPACE" --count)"
echo "WINDOW COUNT: $WINDOW_COUNT"
if [ "$WINDOW_COUNT" -ne 0 ]; then
  echo "- Workspace $WORKSPACE has windows, switching to it"
  aerospace workspace "$WORKSPACE"
  exit 0
fi

echo "Workspace $WORKSPACE has no windows. Creating moving it to current monitor"

MONITOR_ID=$(aerospace list-monitors --focused --format "%{monitor-id}")
# It's unclear if there are two identical monitors. `move-workspace-to-monitor`
# only accepts <monitor-pattern> and not a monitor id
MONITOR_NAME=$(aerospace list-monitors --focused --format "%{monitor-name}")
echo "Current focused monitor: $MONITOR_NAME - $MONITOR_ID"

# For some reasons, move-workspace-to-monitor also moves the other monitors workspace
# to the default one, so we have to first find that workspace to switch to it after the
# workspace move
#
# Get the list of all currently visible workspaces on all monitors
FOCUSED_WORKSPACES=""
for MONITOR in $(aerospace list-monitors --format "%{monitor-id}"); do
  if [ "$MONITOR" = "$MONITOR_ID" ]; then
    echo "Skipping current monitor since we will move the workspace to it"
    continue
  fi
  CURRENT_VISIBLE="$(aerospace list-workspaces --monitor "$MONITOR" --visible)"
  # Make sure its the workspace we want to focus isn't empty on a monitor
  # else it would move it
  if [ "$CURRENT_VISIBLE" = "$WORKSPACE" ]; then
    echo "Monitor is empty and already visible, don't move it"
    exit 0
  fi
  FOCUSED_WORKSPACES="$FOCUSED_WORKSPACES;$CURRENT_VISIBLE"
done

# We could check if the monitor is on the current space, but its easier/quicker to just
# move the workspace to the current monitor even if it might be already there
aerospace move-workspace-to-monitor --workspace "$WORKSPACE" "$MONITOR_NAME"

for WORKSPACE_ID in $(echo "$FOCUSED_WORKSPACES" | tr ';' ' '); do
  # Switch to workspace
  aerospace workspace "$WORKSPACE_ID"
done
aerospace workspace "$WORKSPACE"
