#date: 2021-09-21T17:10:56Z
#url: https://api.github.com/gists/6aceac1361e5af9d2b56d5d3d184f9aa
#owner: https://api.github.com/users/alijmlzd

#!/bin/bash
backupDate=$(date +"%Y%m%d")
databaseUser=DB_USER
databasePass=DB_PASS
databaseName=DB_NAME
databaseHost=127.0.0.1
databasePort=3306
backupTempPath=/path/to/temp/directory
filesPath=/path/to/files/directory
databaseBackupName=$backupTempPath/backups/databaseBackup.sql
filesBackupName=$backupTempPath/backups/filesBackup.tar.gz
fullBackupName=$backupTempPath/fullBackup-$backupDate.tar.gz
scpPath=USER@HOST:/path/to/backups/directory

echo "**************** Create sql backup ********************"
sleep 2
mysqldump -P $databasePort -h $databaseHost -u $databaseUser -p$databasePass $databaseName > $databaseBackupName


echo "**** Archive and gzip files ****"
sleep 2
tar cvzf $filesBackupName -C $filesPath .
chmod +x $filesBackupName

echo "**** Archive and gzip files and database together ****"
sleep 2
tar cvzf $fullBackupName -C $backupTempPath/backups .
chmod +x $fullBackupName


echo "*********** Copy fullBackup to destination ***********"
sleep 2
scp $fullBackupName $scpPath
# Use below instead in the case of ssh key required
# scp -i PUBLIC_KEY $fullBackupName $scpPath


echo "************* Delete files, database and full backup *****************"
rm -rf $databaseBackupName $filesBackupName $fullBackupName