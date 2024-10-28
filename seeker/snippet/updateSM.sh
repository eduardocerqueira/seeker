#date: 2024-10-28T17:00:30Z
#url: https://api.github.com/gists/cecd32d0275ceb558cf86db5aeea7377
#owner: https://api.github.com/users/T313C0mun1s7

#!/bin/bash

# Script to update SmarterMail on an Ubuntu server
# This script automatically scrapes the release notes page to find the latest build number,
# then downloads and installs the latest version of SmarterMail.

# Constants
RELEASE_NOTES_URL="https://www.smartertools.com/smartermail/release-notes/current"
LOG_FILE="/var/log/smartermail_update.log"
MAX_LOG_SIZE=10485760 # 10MB

# Function to log messages
echo_log() {
  local message="$1"
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" | sudo tee -a "$LOG_FILE" > /dev/null
}

# Function to print messages to user
echo_user() {
  local message="$1"
  echo "$message"
}

# Log rotation function
rotate_log() {
  if [[ -f "$LOG_FILE" && $(stat -c%s "$LOG_FILE") -ge $MAX_LOG_SIZE ]]; then
    sudo mv "$LOG_FILE" "$LOG_FILE.old"
    echo_log "Log rotated."
  fi
}

# Function to extract the latest build number
get_latest_build_number() {
  echo_log "Fetching the latest build number from $RELEASE_NOTES_URL..."
  local build_number=$(curl -s "$RELEASE_NOTES_URL" | grep -Po '(?<=<h3 id=")\d{4}(?=" class="secondary-title text-start">Build)' | head -n 1)

  if [[ -z "$build_number" ]]; then
    echo_log "Error: Could not retrieve the latest build number. Exiting."
    echo_user "Error: Could not retrieve the latest build number. Exiting."
    exit 1
  fi

  echo_log "Latest build number retrieved: $build_number"
  echo "$build_number"
}

# Function to download and install the latest SmarterMail
update_smartermail() {
  local build_number="$1"
  local download_url="https://downloads.smartertools.com/smartermail/100.0.$build_number/smartermail_$build_number"

  echo_user "Downloading SmarterMail build $build_number..."
  echo_log "Downloading SmarterMail build $build_number..."
  wget "$download_url" -O "smartermail_$build_number"

  if [[ $? -ne 0 ]]; then
    echo_log "Error: Failed to download SmarterMail build $build_number. Exiting."
    echo_user "Error: Failed to download SmarterMail build $build_number. Exiting."
    exit 1
  fi

  echo_user "Setting executable permissions for the downloaded file..."
  echo_log "Setting executable permissions for the downloaded file..."
  chmod +x "smartermail_$build_number"

  echo_user "Installing SmarterMail build $build_number..."
  echo_log "Installing SmarterMail build $build_number..."
  sudo ./smartermail_$build_number install

  if [[ $? -ne 0 ]]; then
    echo_log "Error: Failed to install SmarterMail build $build_number. Exiting."
    echo_user "Error: Failed to install SmarterMail build $build_number. Exiting."
    exit 1
  fi

  echo_log "SmarterMail build $build_number installed successfully."
  echo_user "SmarterMail build $build_number installed successfully."
}

# Main script
rotate_log
latest_build=$(get_latest_build_number)

# Prompt user for confirmation
echo_user "Do you want to proceed with updating SmarterMail to build $latest_build? (y/n)"
read -r choice
if [[ "$choice" =~ ^[Yy]$ ]]; then
  update_smartermail "$latest_build"
else
  echo_log "Update cancelled by user."
  echo_user "Update cancelled by user."
fi
