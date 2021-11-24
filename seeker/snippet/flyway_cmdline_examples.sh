#date: 2021-11-24T17:11:47Z
#url: https://api.github.com/gists/a0a7bdb83f92cc75781bfd56d5bb90e3
#owner: https://api.github.com/users/hugoheden

# Flyway command line examples
#

# Print some info about the migrations present in the DB relating to some migration-directory 
flyway -X -url='jdbc:mariadb://localhost:3306/my_db?currentSchema=my_schema' \
    -user='my_db_user' -password='my_db_passwd' \
    -locations=filesystem:./core/src/main/resources/db/migration \
    info

# Migrate the DB using the files in the directory, but only up to version 1.2
flyway -X -url='jdbc:mariadb://localhost:3306/my_db?currentSchema=my_schema' \
    -user='my_db_user' -password='my_db_passwd' \
    -locations=filesystem:./core/src/main/resources/db/migration \
    -target='1.2' migrate 