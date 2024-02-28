#date: 2024-02-28T16:55:08Z
#url: https://api.github.com/gists/cee0167ab79079a1c1d4f2ef0f758286
#owner: https://api.github.com/users/wyvern800

# Backup
docker exec CONTAINER /usr/bin/mysqldump -u root --password= "**********"

# Restore
cat backup.sql | docker exec -i CONTAINER /usr/bin/mysql -u root --password= "**********"

ot DATABASE

