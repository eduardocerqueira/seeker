#date: 2021-12-14T17:03:57Z
#url: https://api.github.com/gists/e56c7d752e6aaab297742f86373a0814
#owner: https://api.github.com/users/Yossi1114

#!/bin/bash

# Google Drive Backup Script
# Refer to https://medium.com/swlh/using-rclone-on-linux-to-automate-backups-to-google-drive-d599b49c42e8 and set up rclone first
# Connection to your Google Drive must be established with rclone before upload

now=$(date +"%m%d%Y")
/usr/bin/zip -r backup_$now.zip /var/www/ # backup file/directory path

/usr/bin/rclone copy --update --verbose --transfers 30 --checkers 8 --contimeout 60s --timeout 300s --retries 3 --low-level-retries 10 --stats 1s "backup_$now.zip" "gdrive://"

/usr/bin/rm backup_$now.zip