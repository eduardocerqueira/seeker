#date: 2023-03-23T17:00:12Z
#url: https://api.github.com/gists/f92f26d5dc9b4f0ecf10932951b506c8
#owner: https://api.github.com/users/albertkurnia

#!/bin/bash

# MySQL database credentials
USER="username"
PASSWORD= "**********"
DATABASE="database_name"

# Backup directory
BACKUP_DIR="/path"

# Timestamp (YYYY-MM-DD)
TIMESTAMP=$(date + %F)

# Backup filename
BACKUP_FILENAME="${DATABASE}_${TIMESTAMP}.sql"

# Full backup path
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILENAME}"

# Run mysqldump
mysqldump --user= "**********"=$PASSWORD $DATABASE > $BACKUP_PATH

# Check if backup succeeded
if [ $? -eq 0]; then
  echo "Mysql backup succeeded"
else 
  echo "MySQL backup failed"
fi