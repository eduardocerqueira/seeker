#date: 2025-04-10T16:50:37Z
#url: https://api.github.com/gists/9e4191c9ecdfae59f56685b03e8c048a
#owner: https://api.github.com/users/vikramsoni2

#!/bin/bash

# Configuration
CONFIG_FILE="$HOME/.syncservice.conf"
LOG_FILE="$HOME/sync.log"
CRON_COMMENT="# SYNC_PROJECTS_CRON"

# Load configuration
load_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Configuration file $CONFIG_FILE not found" >&2
        exit 1
    fi
    source "$CONFIG_FILE"
}

# Main sync function
run_sync() {
    # Validate directories
    if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
        echo "Error: Source/Destination directories not configured" >&2
        exit 1
    fi

    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Error: Source directory $SOURCE_DIR does not exist" >&2
        exit 1
    fi

    # Prepare rsync command
    rsync_args=(
        -av
        --no-owner --no-group
        --delete
    )

    # Add exclude patterns
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        rsync_args+=(--exclude="$pattern")
    done

    rsync_args+=("${SOURCE_DIR%/}/" "$DEST_DIR")

    # Create destination directory if needed
    mkdir -p "$DEST_DIR"

    # Execute sync
    echo "Starting sync: $(date)"
    if rsync "${rsync_args[@]}"; then
        echo "Sync completed successfully"
    else
        echo "Sync failed with error code $?" >&2
        exit 1
    fi
}

# Interactive cron setup
setup_cron() {
    echo "Would you like to set up automatic syncing? (yes/no)"
    read -r answer
    
    if [ "$answer" != "yes" ]; then
        echo "Skipping cron setup"
        return
    fi

    # Check for existing cron job
    if crontab -l | grep -qF "$CRON_COMMENT"; then
        echo "Cron job already exists. Skipping setup."
        return
    fi

    # Get schedule selection
    echo "Select sync frequency:"
    echo "1) Every hour"
    echo "2) Every 2 hours"
    echo "3) Every 6 hours"
    echo "4) Daily"
    echo "5) Custom cron schedule"
    echo "6) Cancel"
    read -r -p "Enter choice (1-6): " choice

    case $choice in
        1) schedule="0 * * * *" ;;
        2) schedule="0 */2 * * *" ;;
        3) schedule="0 */6 * * *" ;;
        4) schedule="0 0 * * *" ;;
        5)
            read -r -p "Enter cron schedule (e.g., '0 */4 * * *'): " schedule
            ;;
        6)
            echo "Cron setup cancelled"
            return
            ;;
        *)
            echo "Invalid choice"
            return
            ;;
    esac

    # Get absolute path to script
    script_path=$(realpath "$0")

    # Add to crontab
    (
        crontab -l 2>/dev/null
        echo "$CRON_COMMENT"
        echo "$schedule $script_path --sync >> $LOG_FILE 2>&1"
    ) | crontab -

    echo "Cron job installed successfully!"
    echo "Syncing will run on schedule: $schedule"
}

# Execution flow
if [ "$1" = "--sync" ]; then
    load_config
    run_sync
else
    load_config
    # Interactive mode
    if [ -t 0 ]; then
        setup_cron
        echo -e "\nRunning initial sync..."
    fi
    run_sync
fi
