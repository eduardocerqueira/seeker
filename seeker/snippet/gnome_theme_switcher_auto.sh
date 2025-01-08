#date: 2025-01-08T17:09:15Z
#url: https://api.github.com/gists/425114b11f7b256a29078b9691b00664
#owner: https://api.github.com/users/msanjeevkumar

#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Variables
SCRIPT_NAME="toggle-gnome-theme.sh"
SCRIPT_PATH="/usr/local/bin/$SCRIPT_NAME"
SERVICE_NAME="toggle-gnome-theme"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
TIMER_7AM_FILE="/etc/systemd/system/${SERVICE_NAME}-7am.timer"
TIMER_6PM_FILE="/etc/systemd/system/${SERVICE_NAME}-6pm.timer"

# Improve logging function with severity levels and default message
log() {
    local level="${1:-INFO}"
    local message="${2:-No message provided}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message"
}

# Add error handling function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "ERROR" "Script failed with exit code $exit_code"
    fi
    exit $exit_code
}
trap cleanup EXIT

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
  log "ERROR" "Please run as root"
  exit 1
fi

# Validate input
if [ "$#" -ne 1 ]; then
  log "ERROR" "Usage: $0 {setup|unset}"
  exit 1
fi

# Create the toggle script
create_toggle_script() {
  log "INFO" "Creating toggle script at $SCRIPT_PATH..."

  if ! sudo tee "$SCRIPT_PATH" > /dev/null <<EOF
#!/bin/bash

# Get the current color scheme
CURRENT_SCHEME=\$(gsettings get org.gnome.desktop.interface color-scheme)

# Toggle between 'prefer-dark' and 'default'
if [[ "\$CURRENT_SCHEME" == "'prefer-dark'" ]]; then
  gsettings set org.gnome.desktop.interface color-scheme 'prefer-light'
  echo "Switched to light theme."
else
  gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'
  echo "Switched to dark theme."
fi
EOF
  then
    log "ERROR" "Failed to create toggle script"
    return 1
  fi

  if ! sudo chmod +x "$SCRIPT_PATH"; then
    log "ERROR" "Failed to make script executable"
    return 1
  fi
  log "INFO" "Toggle script created and made executable."
}

# Function to set up the timers
setup_timers() {
  log "INFO" "Setting up timers..."

  if ! create_toggle_script; then
    log "ERROR" "Failed to create toggle script"
    return 1
  fi

  # Create the systemd service file
  sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Toggle GNOME color scheme

[Service]
Type=oneshot
ExecStart=$SCRIPT_PATH

[Install]
WantedBy=multi-user.target
EOF

  # Create the 7 AM timer file
  sudo tee "$TIMER_7AM_FILE" > /dev/null <<EOF
[Unit]
Description=Run GNOME theme toggle script at 7 AM

[Timer]
OnCalendar=*-*-* 07:00:00
Persistent=true
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

  # Create the 6 PM timer file
  sudo tee "$TIMER_6PM_FILE" > /dev/null <<EOF
[Unit]
Description=Run GNOME theme toggle script at 6 PM

[Timer]
OnCalendar=*-*-* 18:00:00
Persistent=true
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF

  # Reload systemd
  if ! sudo systemctl daemon-reload; then
    log "ERROR" "Failed to reload systemd"
    return 1
  fi

  # Enable and start the service first
  if ! sudo systemctl enable "${SERVICE_NAME}.service"; then
    log "ERROR" "Failed to enable service"
    return 1
  fi

  # Enable and start the timers
  if ! sudo systemctl enable --now "${SERVICE_NAME}-7am.timer"; then
    log "ERROR" "Failed to enable 7am timer"
    return 1
  fi
  if ! sudo systemctl enable --now "${SERVICE_NAME}-6pm.timer"; then
    log "ERROR" "Failed to enable 6pm timer"
    return 1
  fi

  log "INFO" "Timers created and enabled. Here are the active timers:"
  systemctl list-timers
}

# Function to unset the timers
unset_timers() {
  log "Unsetting timers..."

  # Disable and stop the timers
  sudo systemctl disable --now "${SERVICE_NAME}-7am.timer"
  sudo systemctl disable --now "${SERVICE_NAME}-6pm.timer"

  # Remove the service and timer files
  sudo rm -f "$SERVICE_FILE" "$TIMER_7AM_FILE" "$TIMER_6PM_FILE"

  # Remove the toggle script
  sudo rm -f "$SCRIPT_PATH"

  # Reload systemd
  sudo systemctl daemon-reload

  log "Timers, service, and toggle script removed."
}

# Main script logic
case "$1" in
  setup)
    setup_timers
    ;;
  unset)
    unset_timers
    ;;
  *)
    log "Usage: $0 {setup|unset}"
    exit 1
    ;;
esac
