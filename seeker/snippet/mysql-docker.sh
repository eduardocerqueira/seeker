#date: 2022-03-22T17:01:16Z
#url: https://api.github.com/gists/5f49929cf160f2d4b6f3b2737d2bcc50
#owner: https://api.github.com/users/lucatsf

# Backup
docker exec CONTAINER /usr/bin/mysqldump -u root --password=root DATABASE > dump.sql

# Restore
cat dump.sql | docker exec -i CONTAINER /usr/bin/mysql -u root --password=root DATABASE

