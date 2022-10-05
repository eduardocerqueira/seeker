#date: 2022-10-05T17:16:34Z
#url: https://api.github.com/gists/2e3272d4cccf99137e419939c7c80b9d
#owner: https://api.github.com/users/herickcamara

# Backup
docker exec CONTAINER /usr/bin/mysqldump -u root --password= "**********"

# Restore
cat backup.sql | docker exec -i CONTAINER /usr/bin/mysql -u root --password= "**********"

ot DATABASE

