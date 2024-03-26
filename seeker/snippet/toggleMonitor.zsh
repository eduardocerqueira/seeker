#date: 2024-03-26T16:54:15Z
#url: https://api.github.com/gists/ecfa471defc86e7a71f3d4b1a6be1759
#owner: https://api.github.com/users/sbasegmez

#!/bin/zsh

# Script to toggle between two displays for Mac M1 on DDC supported monitor
#
# Install m1ddc first (brew install m1ddc)
# Works on Mac M1
# Developed by Serdar Basegmez shared with WTFPL

# Config:
DDC_COMMAND="/opt/homebrew/bin/m1ddc"
SAY_COMMAND="/usr/bin/say"
TMP_FILE="/Users/$USER/.lastMonitorInput"
DISPLAY_NO=1
INPUT_ID_1=15
INPUT_ID_2=0 # We made this automatic. USB-C is 27 

# Because the get input command takes time to be updated on the DDC, it's best to store the last value in a file.
if [[ -f $TMP_FILE ]]; then
    DISPLAY_LAST_INPUT=$(cat $TMP_FILE)
else
    # Fail if the file doesn't exist
    DISPLAY_LAST_INPUT=$($DDC_COMMAND display $DISPLAY_NO get input)
fi

if [[ $DISPLAY_LAST_INPUT == $INPUT_ID_1 ]]; then
    NEW_INPUT=$INPUT_ID_2
else
    NEW_INPUT=$INPUT_ID_1
fi

$DDC_COMMAND display $DISPLAY_NO set input $NEW_INPUT

# Save the current value of DISPLAY_LAST_INPUT into the file named in TMP_FILE.
# This will be used in the next script invocation to know the last state of the display.
echo "$NEW_INPUT" > "$TMP_FILE"

# We do audio feedback here as the screen is probably gone by now.
$SAY_COMMAND "Display set to $NEW_INPUT"


