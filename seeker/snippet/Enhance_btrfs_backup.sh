#date: 2025-02-03T16:57:56Z
#url: https://api.github.com/gists/ae6e623923d39873f84f473b7dcad605
#owner: https://api.github.com/users/greatplr

#!/bin/bash

################################################################################
# BTRFS Backup Script for Enhance Backup Server
#
# Overview:
# This script facilitates the efficient backup of website data stored on an 
# Enhance Backup Server utilizing the BTRFS filesystem. It performs the 
# following operations for each website directory:
#   1. Synchronizes the live state (`backup-subvolume`) to the destination server.
#   2. Sends all BTRFS snapshots (`snapshot-*`) to the destination server using
#      `btrfs send` and `btrfs receive`.
#   3. Deletes outdated snapshots from the destination server that no longer 
#      exist on the source server.
#
# Key Features:
# - Deletes old snapshots from the destination server to maintain sync with 
#   the source.
# - Skips transferring snapshots that are already present on the destination.
# - Temporarily sets writable subvolumes and snapshots to read-only during 
#   transfer to ensure compatibility with `btrfs send`.
# - Automatically restores writable status to subvolumes and snapshots after 
#   the transfer.
# - Logs both successful and failed operations for auditing and debugging.
# - **Restore Mode**: A special option for restoring backups when your Enhance 
#   Backup server has failed. In this scenario:
#     1. Set `restore_mode=on` in the script.
#     2. Run the script in the **opposite direction**, transferring data from 
#        the backup server (destination) back to the main Enhance Backup server (source).
#     3. The script will ensure all transferred subvolumes and snapshots are set 
#        to writable (`ro=false`), making the backups ready for immediate use.
#
# Requirements:
# - Both the source and destination `/backups` directories must reside on a BTRFS filesystem.
# - SSH access between the source and destination servers.
# - Sufficient privileges to manage BTRFS subvolumes on both servers.
#
# Logs:
# - Successful operations: /var/log/btrfs-backup.log
# - Errors and failures:   /var/log/btrfs-backup-errors.log
#
# Disclaimer:
# THIS SCRIPT IS PROVIDED "AS IS" WITHOUT ANY WARRANTY. USE AT YOUR OWN RISK.
# Ensure you test this script in a non-production environment before deploying.
# Linkers Gate LLC (cPFence.app) assumes no responsibility for any damage, 
# data loss, or other issues caused by the use of this script.
################################################################################

# Paths
SOURCE_BACKUPS="/backups"                 # Path to backups on the source server
DEST_SERVER_IP="168.162.11.11"          # Destination server IP address
DEST_SERVER="root@$DEST_SERVER_IP"        # Destination server SSH credentials
SOURCE_HOSTNAME=$(hostname)                 # Gets the hostname of the source server
DEST_BACKUPS="/backups/$SOURCE_HOSTNAME"  # Path to backups on the destination server - Store backups in a seperate folder for each source backup server

# Logs
LOG_FILE="/var/log/btrfs-backup.log"     # Log file for successful operations
ERROR_LOG_FILE="/var/log/btrfs-backup-errors.log"  # Log file for errors and failures

# Restore Mode
# Set to "on" to make subvolumes and snapshots writable during restoration
restore_mode="off"  

# Ensure the log files exist
touch "$LOG_FILE"
touch "$ERROR_LOG_FILE"

################################################################################
# MAIN BACKUP PROCESS
################################################################################

# Process each website directory in the source backup folder
for website_dir in "$SOURCE_BACKUPS"/*/; do
    echo "[$SOURCE_HOSTNAME] Processing website directory: $website_dir"

    # Ensure the destination directory exists
    ssh "$DEST_SERVER" "sudo mkdir -p $DEST_BACKUPS/$(basename $website_dir)"

    ### 1. Synchronize the live state (backup-subvolume)
    if [ -d "$website_dir/backup-subvolume" ]; then
        echo "Processing backup-subvolume: $website_dir/backup-subvolume"

        # Temporarily set backup-subvolume to read-only
        subvolume_readonly=$(sudo btrfs property get "$website_dir/backup-subvolume" ro | grep -q "true"; echo $?)
        if [ "$subvolume_readonly" -ne 0 ]; then
            echo "Setting $website_dir/backup-subvolume to read-only..."
            sudo btrfs property set -f "$website_dir/backup-subvolume" ro true
            subvolume_readonly_set=true
        else
            subvolume_readonly_set=false
        fi

        # Delete the existing backup-subvolume on the destination if it exists
        resolved_subvolume=$(readlink -f "$website_dir/backup-subvolume")
        echo "Checking for existing backup-subvolume on the destination..."
        if ssh "$DEST_SERVER" "sudo test -d $DEST_BACKUPS/$(basename "$website_dir")/backup-subvolume"; then
            echo "Existing backup-subvolume found. Deleting on destination..."
            ssh "$DEST_SERVER" "sudo btrfs subvolume delete $DEST_BACKUPS/$(basename "$website_dir")/backup-subvolume"
        fi

        # Send the updated backup-subvolume
        echo "Transferring backup-subvolume: $resolved_subvolume"
        if sudo btrfs send "$resolved_subvolume" | ssh "$DEST_SERVER" "sudo btrfs receive $DEST_BACKUPS/$(basename "$website_dir")"; then
            echo "$resolved_subvolume" >> "$LOG_FILE"
            echo "Backup-subvolume $resolved_subvolume transferred successfully."

            # Set to writable if restore_mode is on
            if [ "$restore_mode" == "on" ]; then
                echo "Setting $DEST_BACKUPS/$(basename "$website_dir")/backup-subvolume to writable (ro=false)..."
                ssh "$DEST_SERVER" "sudo btrfs property set -f $DEST_BACKUPS/$(basename "$website_dir")/backup-subvolume ro false"
            fi
        else
            echo "Error: Failed to transfer backup-subvolume $resolved_subvolume." | tee -a "$ERROR_LOG_FILE"
        fi

        # Restore backup-subvolume to writable if necessary
        if [ "$subvolume_readonly_set" = true ]; then
            echo "Restoring $website_dir/backup-subvolume to writable..."
            sudo btrfs property set -f "$website_dir/backup-subvolume" ro false
        fi
    else
        echo "Warning: $website_dir/backup-subvolume does not exist or is not a directory. Skipping..."
    fi

    ### 2. Synchronize and manage snapshots
    # Retrieve snapshot lists from the source and destination
    source_snapshots=$(find "$website_dir" -maxdepth 1 -type d -name "snapshot-*" -printf "%f\n" | sort)
    dest_snapshots=$(ssh "$DEST_SERVER" "find $DEST_BACKUPS/$(basename "$website_dir") -maxdepth 1 -type d -name 'snapshot-*' -printf '%f\n' | sort")

    # Delete snapshots on the destination that are no longer present on the source
    for dest_snapshot in $dest_snapshots; do
        if ! echo "$source_snapshots" | grep -q "^$dest_snapshot$"; then
            echo "Deleting outdated snapshot on destination: $dest_snapshot"
            ssh "$DEST_SERVER" "sudo btrfs subvolume delete $DEST_BACKUPS/$(basename "$website_dir")/$dest_snapshot"
        fi
    done

    # Send new snapshots from the source to the destination
    for snapshot in "$website_dir"/snapshot-*; do
        if [ -d "$snapshot" ]; then
            snapshot_name=$(basename "$snapshot")

            # Skip snapshots that already exist on the destination
            if echo "$dest_snapshots" | grep -q "^$snapshot_name$"; then
                echo "Snapshot $snapshot_name already exists on the destination. Skipping..."
                continue
            fi

            # Temporarily set the snapshot to read-only
            snapshot_readonly=$(sudo btrfs subvolume show "$snapshot" | grep -q "Flags: readonly"; echo $?)
            if [ "$snapshot_readonly" -ne 0 ]; then
                echo "Setting snapshot $snapshot to read-only..."
                sudo btrfs property set -f "$snapshot" ro true
                readonly_set=true
            else
                readonly_set=false
            fi

            # Transfer the snapshot
            echo "Transferring snapshot: $snapshot"
            if sudo btrfs send "$snapshot" | ssh "$DEST_SERVER" "sudo btrfs receive $DEST_BACKUPS/$(basename "$website_dir")"; then
                echo "$snapshot" >> "$LOG_FILE"
                echo "Snapshot $snapshot transferred successfully."

                # Set to writable if restore_mode is on
                if [ "$restore_mode" == "on" ]; then
                    echo "Setting $DEST_BACKUPS/$(basename "$website_dir")/$snapshot_name to writable (ro=false)..."
                    ssh "$DEST_SERVER" "sudo btrfs property set -f $DEST_BACKUPS/$(basename "$website_dir")/$snapshot_name ro false"
                fi
            else
                echo "Error: Failed to transfer snapshot $snapshot." | tee -a "$ERROR_LOG_FILE"
            fi

            # Restore writable status for the snapshot if necessary
            if [ "$readonly_set" = true ]; then
                echo "Restoring $snapshot to writable..."
                sudo btrfs property set -f "$snapshot" ro false
            fi
        fi
    done
done


#####################################
#       Tips & Troubleshooting:     #
#####################################
#
# Sometimes if this script gets halted or stopped unexpectedly, some btrfs snapshots are left in read-only status.
# This can cause issues with Enhance backup/restore functionalities. To fix this, you can manually search and set them back to writable.
#
# To search, use this command:
#
# for subvolume in /backups/*/backup-subvolume /backups/*/snapshot-*; do
#     if [ -e "$subvolume" ]; then
#         ro_status=$(sudo btrfs property get -t subvol "$subvolume" ro 2>/dev/null | awk -F'=' '/ro/ {print $2}')
#         if [ "$ro_status" == "true" ]; then
#             echo "$subvolume: ro=true"
#         fi
#     fi
# done
#
# To search and fix, use this command:
#
# for subvolume in /backups/*/backup-subvolume /backups/*/snapshot-*; do
#     if [ -e "$subvolume" ]; then
#         ro_status=$(sudo btrfs property get -t subvol "$subvolume" ro 2>/dev/null | awk -F'=' '/ro/ {print $2}')
#         if [ "$ro_status" == "true" ]; then
#             echo "Setting $subvolume to writable (ro=false)..."
#             sudo btrfs property set -f -t subvol "$subvolume" ro false || {
#                 echo "Failed to set $subvolume to writable."
#             }
#         fi
#     fi
# done
#
#
###############################
# To quickly delete all snapshots on a backup server (use with caution!)
#
#for subvolume in /backups/*/backup-subvolume /backups/*/snapshot-*; do
#    if [ -d "$subvolume" ]; then
#       echo "Deleting subvolume: $subvolume"
#       if ! sudo btrfs subvolume delete -c "$subvolume"; then
#           echo "Failed to delete subvolume: $subvolume" >&2
#       fi
#   fi
#done
#