#date: 2023-11-10T16:36:00Z
#url: https://api.github.com/gists/515eb7ec704968e691a721e67cfa65f8
#owner: https://api.github.com/users/sebastiancarlos

#!/usr/bin/env bash

# All my gist code is licensed under the MIT license.

# Add this to your PATH

# if calling without arguments or with -h or --help, print usage
if [[ $# -eq 0 ]] || [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    echo "Usage: get-terminal-cwd <terminal-pid>"
    echo "  - return the CWD of the terminal with the given PID"
    echo "  - this is done by checking the first child of the terminal, which is usually a shell"
    echo "  - default to ~ if there is no child"
    exit 1
fi

# get pid of first child of terminal
first_child=$(command pgrep -P $1 | head -n 1)

# if there is no child, default to $HOME
if [[ -z "${first_child}" ]]; then
  echo "${HOME}"
else
  echo $(readlink -f /proc/${first_child}/cwd)
fi