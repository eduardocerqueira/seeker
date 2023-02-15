#date: 2023-02-15T16:46:49Z
#url: https://api.github.com/gists/b49b6268d4ebb2e4f026641a4a157d24
#owner: https://api.github.com/users/phpcoinn

cd /var/www/phpcoin

# export existing database
mysqldump phpcoin > phpcoin.sql

# stop services
service mysql stop
service apache2 stop

# remove mysql server completely
apt update
apt purge mysql-server* -y
apt purge mysql-client-* -y
apt autoremove -y
apt autoclean
deluser mysql
rm -rf /var/lib/mysql
rm -rf /etc/mysql

# this must be executed in order to install later mariadb server
echo "/usr/sbin/mysqld { }" > /etc/apparmor.d/usr.sbin.mysqld
apparmor_parser -v -R /etc/apparmor.d/usr.sbin.mysqld
rm /etc/apparmor.d/usr.sbin.mysqld

# install mariadb server
apt install mariadb-server -y

# setuo user, database and privileges
export DB_NAME=phpcoin
export DB_USER=phpcoin
export DB_PASS=phpcoin

mysql -e "create database $DB_NAME;"
mysql -e "create user '$DB_USER'@'localhost' identified by '$DB_PASS';"
mysql -e "grant all privileges on $DB_NAME.* to '$DB_USER'@'localhost';"

# convert exported file to correct encoding
sed -i 's/utf8mb4_0900_ai_ci/utf8mb4_general_ci/g' phpcoin.sql

# import database back
mysql phpcoin < phpcoin.sql

# start service
service apache2 start