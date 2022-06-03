#date: 2022-06-03T16:47:02Z
#url: https://api.github.com/gists/d234e556463da770b7083ab3a4336240
#owner: https://api.github.com/users/diogobruno

#!/bin/bash
##
# Get a list of all databases except the system databases that
are not needed
##
DATABASES=$(echo "show databases;" | mysql | grep -Ev "(Database|information_schema|mysql|performance_schema)")
DATE=$(date +%d%m%Y)
TIME=$(date +%s)
BACKUP_DIR=/home/your_user/backup
##
# Create Backup Directory
##
if [ ! -d ${BACKUP_DIR} ]; then
mkdir -p ${BACKUP_DIR}
fi
##
# Backup all databases
##
for DB in $DATABASES;
do
mysqldump --single-transaction --skip-lock-tables $DB | gzip > ${BACKUP_DIR}/$DATE-$DB.sql.gz
done