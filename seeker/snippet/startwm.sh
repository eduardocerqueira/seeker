#date: 2025-09-04T16:50:22Z
#url: https://api.github.com/gists/1d754dfcf4c1fe54462d7a2b047d9bfd
#owner: https://api.github.com/users/alfiyansys

# File location: /etc/xrdp/startwm.sh
# ADDITIONAL LINE BEFORE XRDP Xsession startup execution
# Needs to 'kill' the existing session GUI, or will just closes GUI when accessed via RDP
# Doesn't kill tty sessions
# Based on Gemini discussions: https://g.co/gemini/share/ec5dacdd6a69

USER_TO_LOGOUT=$USER

# Check if loginctl is available. This is crucial for a robust script.
if command -v loginctl >/dev/null 2>&1; then

    # Get a list of sessions for the user and iterate through them
    loginctl list-sessions --no-legend --no-header | while read -r SESSION_ID PID TYPE REST; do
        # Check if the session is a graphical one (X11 or Wayland)
        if [ "$TYPE" = "x11" ] || [ "$TYPE" = "wayland" ]; then
            # Check if this graphical session belongs to the current user
            SESSION_USER=$(loginctl show-session -p User --value "$SESSION_ID")
            if [ "$SESSION_USER" = "$USER_TO_LOGOUT" ]; then
                echo "Found and terminating graphical session ($SESSION_ID) of type $TYPE for user $USER_TO_LOGOUT"
                loginctl terminate-session "$SESSION_ID"
            fi
        fi
    done

    # Give the system a moment to clean up after the termination
    sleep 2
fi

## original lines, xorg exec