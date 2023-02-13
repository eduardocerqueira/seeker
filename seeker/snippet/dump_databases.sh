#date: 2023-02-13T17:11:17Z
#url: https://api.github.com/gists/b2eda31a7d17bc511ea2618bfba5674a
#owner: https://api.github.com/users/Rufmord

#!/bin/sh
backup_directory="your/path/to/backups"
# All running containers have the name database-application

# Postgres
for container in $(docker ps --format "{{.Names}}"| grep postgres | sed 's/ .*//')
do
        docker exec $container pg_dump -U ${container#"postgres-"} > $backup_directory/$container.dump
done
# MariaDB
for container in $(docker ps --format "{{.Names}}"| grep mariadb | sed 's/ .*//')
do
        db_password= "**********"
        docker exec $container mysqldump --user= "**********"=${db_password#"MYSQL_PASSWORD="} ${container#"mariadb-"} > $backup_directory/$container.dump
donetainer#"mariadb-"} > $backup_directory/$container.dump
done