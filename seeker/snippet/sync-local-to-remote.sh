#date: 2025-11-19T17:12:16Z
#url: https://api.github.com/gists/2e078cfc336d6b6a36cd4b9ce19fcf64
#owner: https://api.github.com/users/rahulbarmann

#!/bin/bash

# --- File Synchronization Setup Script ---
# This script sets up a continuous, one-way rsync synchronization 
# from your local macOS machine to a remote Arch Linux VM upon file changes,
# using fswatch.

# Ensure fswatch is installed
if ! command -v fswatch &> /dev/null
then
    echo "Error: fswatch is not installed. Please install it using 'brew install fswatch'."
    exit 1
fi

echo "--- Continuous Development Sync Setup ---"

# 1. Prompt for Remote SSH Details
read -p "Enter Remote Host (user@ip, e.g., rahul@192.168.1.8): " REMOTE_HOST

# 2. Prompt for Local Directory to Monitor
read -p "Enter Local Codebase Path (Source, e.g., ~/projects/assistant): " LOCAL_PATH

# 3. Prompt for Remote Destination Directory
read -p "Enter Remote Destination Path (e.g., /home/rahul/projects/assistant): " REMOTE_PATH

# Validate local path
if [ ! -d "$LOCAL_PATH" ]; then
    echo "Error: Local path '$LOCAL_PATH' does not exist or is not a directory."
    exit 1
fi

# Define the rsync command arguments
# -a: archive mode (preserves permissions, etc.)
# -v: verbose output
# -z: compress file data during the transfer
# --delete: deletes files in the remote directory that no longer exist locally (creates an exact mirror)
RSYNC_ARGS="-avz --delete"

# Define common exclusion folders/files
RSYNC_EXCLUDES="--exclude '.git' --exclude 'node_modules/' --exclude 'target/' --exclude 'venv/' --exclude '__pycache__/' --exclude '*.pyc'"

echo ""
echo "Starting continuous one-way sync..."
echo "Source: $LOCAL_PATH -> Destination: $REMOTE_HOST:$REMOTE_PATH"
echo "Press Ctrl+C to stop the synchronization."
echo ""

# Change to the local directory, which simplifies the rsync source path to './'
cd "$LOCAL_PATH" || exit

# Run fswatch:
# -o: one-per-batch, buffers events and runs the command only once per burst of changes
# .: monitor the current directory (which is now $LOCAL_PATH)
# The output is piped to xargs to execute the rsync command
fswatch -o . | xargs -n1 -I{} rsync $RSYNC_ARGS $RSYNC_EXCLUDES ./ "$REMOTE_HOST:$REMOTE_PATH"