#date: 2024-01-25T17:06:06Z
#url: https://api.github.com/gists/61cd1f31dccbae9a92e8f5658af79473
#owner: https://api.github.com/users/dazeb

#!/bin/bash

# Check if we are running inside a screen session
if [ -z "$STY" ]; then
    # Check if screen is installed
    if ! command -v screen &> /dev/null; then
        echo "Screen is not installed. Would you like to install it now? (y/n)"
        read -r install_screen
        if [[ $install_screen == "y" ]]; then
            # Install screen (Debian/Ubuntu example)
            sudo apt-get install screen -y
        else
            echo "Screen is required to safely run this script."
            exit 1
        fi
    fi

    # Start a new screen session with a unique name based on the current timestamp
    SESSION_NAME="script_session_$(date +%s)"
    screen -dmS "$SESSION_NAME" "$0" "$@"
    echo "Started a new screen session named $SESSION_NAME"
    echo "If you get disconnected, log back in and type 'screen -r $SESSION_NAME' to reattach."
    exit 0
fi

# The rest of your script goes here