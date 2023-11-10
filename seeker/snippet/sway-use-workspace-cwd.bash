#date: 2023-11-10T16:36:00Z
#url: https://api.github.com/gists/515eb7ec704968e691a721e67cfa65f8
#owner: https://api.github.com/users/sebastiancarlos

#!/usr/bin/env bash

# All my gist code is licensed under the MIT license.

# Add this to your ~/.bashrc

# sway-use-workspace-cwd
# - this must be implemented as a function because it calls 'cd'
# - usecase: matching scracthpad's cwd to the cwd of the workspace
function sway-use-workspace-cwd () {
  # print usage with -h/--help
  if [[ "$#" -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
      echo "Usage: sway-use-workspace-cwd [workspace-name]"
      echo "  - Change to the CWD of the first window in the provided workspace"
      echo "  - If no workspace is provided, uses the currently focused workspace"
      echo "  - If there is no window in the workspace, change to ~"
      exit 1
  fi

  cd $(sway-get-workspace-cwd "$@")
}