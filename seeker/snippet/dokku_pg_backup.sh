#date: 2025-05-20T17:12:26Z
#url: https://api.github.com/gists/ce2677c85b6b951411a427b98d56d9df
#owner: https://api.github.com/users/israelst

#!/bin/bash

# ----------------------------------------
# Dokku PostgreSQL Backup Script
# ----------------------------------------
# This script loops through all Dokku Postgres services,
# exports each database, compresses the output, and saves it
# to a timestamped file in a dedicated backup directory.
#
# Author: Israel Teixeira <israelst@gmail.com>
# ----------------------------------------

# Exit on any error
set -e

# Set backup directory
BACKUP_DIR="$HOME/dokku-pg-backups"

# Create the backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Get the current date in YYYY-MM-DD format
DATE=$(date +%F)

# List all Dokku Postgres services quietly (just the names)
SERVICES=$(dokku postgres:list --quiet)

echo "Starting backup of Dokku PostgreSQL databases..."
echo "Backup directory: $BACKUP_DIR"
echo "Date: $DATE"
echo

# Loop over each Postgres service and back it up
for SERVICE in $SERVICES; do
    echo "Backing up service: $SERVICE"
    
    # Export the database and compress the output
    dokku postgres:export "$SERVICE" | gzip > "$BACKUP_DIR/${SERVICE}-${DATE}.sql.gz"
    
    echo "Saved: $BACKUP_DIR/${SERVICE}-${DATE}.sql.gz"
    echo
done

echo "✅ All backups completed successfully."
