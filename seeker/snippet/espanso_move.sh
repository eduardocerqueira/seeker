#date: 2025-01-31T16:48:39Z
#url: https://api.github.com/gists/db3ed545deede9cd2cd29ec4fee353c8
#owner: https://api.github.com/users/smeech

#!/bin/bash

# A slightly more efficient version of espanso_move.sh which utilises xwininfo.

# Espanso doesn't facilitate repositioning of its Search or Choice etc. windows.
# If the following script is running in the background, it will await the opening
# of the window, and move it to the top left of the screen.
# Adjust the middle two coordinates (,0,0,) for a different position, and reduce 
# the lower sleep value (included to reduce CPU load) if it's too slow

WINDOW_TITLE="espanso"

# Function to check if the window exists and return window information
get_window_info() { xwininfo -name "$WINDOW_TITLE" 2> /dev/null; }

while true; do
  # Get window information using the function
  WINDOW_INFO=$(get_window_info)
  if [ -n "$WINDOW_INFO" ]; then
    WINDOW_ID=$(echo "$WINDOW_INFO" | awk '/Window id:/ {print $4}')
    wmctrl -i -r "$WINDOW_ID" -e 0,0,0,-1,-1 # gravity, coordinates, resize
    echo "Window '$WINDOW_TITLE' moved."
    # Wait until the window closes
    while [ -n "$(get_window_info)" ]; do sleep 1; done 
    echo "Window '$WINDOW_TITLE' closed or minimized. Waiting for it to reappear."
  fi
  sleep 0.5  # Control polling frequency
done