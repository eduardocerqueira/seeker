#date: 2024-10-03T16:52:40Z
#url: https://api.github.com/gists/9c51b8b3f1631acda07627332d690b4f
#owner: https://api.github.com/users/kovkor

#!/bin/bash

# crontab for daily backup
# 0 3 * * * /home/user/envbackup.sh >> /home/user/backup/backup.log 2>&1


# Backup destination
BACKUP_DIR="/home/user/backup"

# Work directory
WORK_DIR="/home/user/Work"

# Create a timestamp for the backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create a directory for today's backup
TODAY_BACKUP_DIR="$BACKUP_DIR/$TIMESTAMP"

# Check if directories exist
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory does not exist. Creating it now."
    mkdir -p "$BACKUP_DIR"
fi

if [ ! -d "$WORK_DIR" ]; then
    echo "Work directory does not exist. Please check the path."
    exit 1
fi

mkdir -p "$TODAY_BACKUP_DIR"

echo "Starting backup process..."
echo "Searching for .env files in $WORK_DIR"

# Use find command to locate .env files
env_files=$(find "$WORK_DIR" -type f -name ".env*")

# Check if any files were found
if [ -z "$env_files" ]; then
    echo "No .env files found in $WORK_DIR"
    rm -rf "$TODAY_BACKUP_DIR"
    exit 1
fi

# Copy the found files to the backup directory
echo "Copying .env files to backup directory..."
while IFS= read -r file; do
    rel_path="${file#$WORK_DIR/}"
    backup_path="$TODAY_BACKUP_DIR/$rel_path"
    mkdir -p "$(dirname "$backup_path")"
    cp "$file" "$backup_path"
    echo "Backed up: $rel_path"
done <<< "$env_files"

# List all backed up files for verification
echo "Files backed up:"
find "$TODAY_BACKUP_DIR" -type f

# Compress the backup
echo "Compressing backup..."
tar -czvf "$BACKUP_DIR/env_backup_$TIMESTAMP.tar.gz" -C "$BACKUP_DIR" "$TIMESTAMP"

# Verify the contents of the tar file
echo "Contents of the tar file:"
tar -tvf "$BACKUP_DIR/env_backup_$TIMESTAMP.tar.gz"

# Remove the uncompressed backup directory
rm -rf "$TODAY_BACKUP_DIR"

# Keep only the last 30 days of backups
find "$BACKUP_DIR" -name "env_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed successfully. Backup file: $BACKUP_DIR/env_backup_$TIMESTAMP.tar.gz"
