#date: 2023-06-30T17:05:29Z
#url: https://api.github.com/gists/63b7c5dc450ccc2ea93055f95221f6a8
#owner: https://api.github.com/users/juanrgon

#!/bin/bash

# live-tail.sh
# This script runs a provided command and continuously displays the last 10 lines of its output.
# It stops running and clears the corresponding lines once the command finishes execution.
# The purpose is to have a clean terminal output for long-running commands that produce a lot of output.
# Note: This script uses ANSI escape codes to manipulate the cursor and the terminal screen,
# which might not work correctly in all terminal emulators.

# Usage: ./live-tail.sh <command>
# Example: ./live-tail.sh ping google.com

# check if command line arguments are provided
if [ $# -eq 0 ]; then
    echo "No command provided. Usage: $0 <command>"
    exit 1
fi

# start the command and send its output to a file
"$@" > output.txt 2>&1 &
pid=$!

# save cursor position
printf "\0337"

# continuously show the last 10 lines of output
old_output=""
while true; do
  # check if the process is still running
  if ! kill -0 $pid 2> /dev/null; then
    printf "\nCommand finished\n"
    exit 0
  fi
  
  output="$(tail -n 10 output.txt)"
  
  if [ "$output" != "$old_output" ]; then
    # restore to saved cursor position, clear below, save cursor position again
    printf "\0338\033[J\0337"
    echo "$output"
    old_output="$output"
  fi
  
  sleep 1
done
