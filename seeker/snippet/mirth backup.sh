#date: 2022-08-01T16:55:46Z
#url: https://api.github.com/gists/ba864575bd6739ffe80df6a4ff9c27c7
#owner: https://api.github.com/users/ssrowe

#!/bin/bash

# settings the paths
mirthPath="/opt/mirthconnect/"
backupPath="/home/mirth/backups/"
scriptPath=`pwd`"/"
vDate=`date --date 'yesterday' +%Y-%m-%d`

# generating console command
echo exportcfg ${backupPath}mirth_backup-${vDate}.xml > ${scriptPath}backup_cmds

# run backup
${mirthPath}Mirth\ Connect\ CLI -s ${scriptPath}backup_cmds

# generating checksum
cd ${backupPath}
md5sum mirth_backup-${vDate}.xml > mirth_backup-${vDate}.md5
md5sum mirth_backup-${vDate}.xml >> checksums.md5

# compress backup with 7z
7z a -t7z -mx=9 ${backupPath}mirth_backup-${vDate} ${backupPath}mirth_backup-${vDate}.* && rm ${backupPath}mirth_backup-${vDate}.xml ${backupPath}mirth_backup-${vDate}.md5