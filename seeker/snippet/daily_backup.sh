#date: 2025-04-03T17:10:09Z
#url: https://api.github.com/gists/6a3eef284557230ae851660de2c02408
#owner: https://api.github.com/users/leviself56

#!/bin/bash
GIVENNAME="alrb-api2"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
SRC="/var/www/"

REMOTE_SSH_USER="user"
REMOTE_SSH_IP="192.168.23.2"
REMOTE_SSH_DIR="/home/user/backups/alrb-api2/"

MARIADB_USER="user"
MARIADB_PASS="pass"

cd ~
tar -czvf $GIVENNAME-$TIMESTAMP.tar.gz $SRC
scp $GIVENNAME-$TIMESTAMP.tar.gz $REMOTE_SSH_USER@$REMOTE_SSH_IP:$REMOTE_SSH_DIR
rm $GIVENNAME-$TIMESTAMP.tar.gz

mariadb-dump -u $MARIADB_USER -p$MARIADB_PASS -x -A > ~/$GIVENNAME-$TIMESTAMP-databases.sql
scp $GIVENNAME-$TIMESTAMP-databases.sql $REMOTE_SSH_USER@$REMOTE_SSH_IP:$REMOTE_SSH_DIR
rm $GIVENNAME-$TIMESTAMP-databases.sql