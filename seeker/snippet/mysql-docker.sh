#date: 2022-09-01T16:59:50Z
#url: https://api.github.com/gists/9a7afd304e9426a5e6bde4a2c064ffef
#owner: https://api.github.com/users/brianraila

# Backup
docker exec CONTAINER /usr/bin/mysqldump -u root --password= "**********"

# Restore
cat backup.sql | docker exec -i CONTAINER /usr/bin/mysql -u root --password= "**********"

ot DATABASE

