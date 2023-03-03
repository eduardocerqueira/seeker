#date: 2023-03-03T16:49:21Z
#url: https://api.github.com/gists/927a800e771e3573c47ffa9d925ec5c2
#owner: https://api.github.com/users/Hrumble

#!/bin/bash

# Define backup directory path
backup_dir=/path/to/backup/directory

mkdir -p $backup_dir

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

echo "Enter files and directories to backup (separated by spaces):"
read -a files_to_backup

# Create backup archive
backup_file="$backup_dir/backup_$timestamp.tar.gz"
tar -czvf $backup_file "${files_to_backup[@]}"

echo "Backup created: $backup_file"