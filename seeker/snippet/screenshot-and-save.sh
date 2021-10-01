#date: 2021-10-01T01:26:44Z
#url: https://api.github.com/gists/4252cb0e7cce086e9885da1a3b78b0fb
#owner: https://api.github.com/users/rhian-cs

#!/bin/bash

#####    DESCRIPTION    #####
# screenshot-and-save.sh - version 3.0.0
#
# This script takes a screenshot and automatically saves it to
# a desired location.
#
# You can configure it and then bind it to a key or the script
# it directly.
#
# Call `./screenshot-and-save.sh --help` for usage or see the `print_usage_and_exit` function below
#
#####    DEPENDENCIES   #####
#
# The `scrot` executable must be present for this script to work.
#
#####   CONFIGURATION   #####

SCREENSHOTS_DIR="${HOME}/Pictures/Screenshots"
DATE=$(date +'%Y-%m-%d')
HOUR=$(date +'%H-%M-%S')
FILENAME="Screenshot from ${DATE} ${HOUR}.png"

#####   -------------   #####

PROGRAM_NAME=$0
OPERATION=$1

function print_usage_and_exit {
  echo "Usage: $PROGRAM_NAME [OPERATION]"
  echo "Operations:"
  echo "  $PROGRAM_NAME --full    Screenshot entire screen (including other monitors, if present)"
  echo "  $PROGRAM_NAME --window-with-border  Screenshot current focused screen (window borders included)"
  echo "  $PROGRAM_NAME --help    Display this message"

  exit $1
}

if [[ $OPERATION == "--window-with-border" ]]; then
  # Scrot options:
  # -b -> Grab window border
  # -u -> Print currently focused window
  # -e -> Execute a command afterwards
	SCROT_OPTIONS="-bue"

elif [[ $OPERATION == "--full" ]]; then
  # Scrot options:
  # -e -> Execute a command afterwards
  SCROT_OPTIONS="-e"

elif [[ $OPERATION == "--help" ]]; then
  print_usage_and_exit 0
else
  print_usage_and_exit 1
fi

mkdir -p "$SCREENSHOTS_DIR" # Create directory if it doesn't exist
FILE_OUTPUT_PATH="$SCREENSHOTS_DIR/$FILENAME"

# Take a screenshot with the desired options and move it to the desired path
scrot "$SCROT_OPTIONS" "mv \$f '$FILE_OUTPUT_PATH'"