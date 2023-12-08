#date: 2023-12-08T17:06:06Z
#url: https://api.github.com/gists/f412253cc79a20b5bacd2eb7703c30b8
#owner: https://api.github.com/users/bzxxxxxx

# Completely remove any previous config
sudo apt remove --purge mysql*
sudo apt autoremove
sudo find / -iname mysql

# install the server
sudo apt update
sudo apt install mysql-server
# run the wizard
sudo mysql_secure_installation
sudo mysql
mysql> use mysql;
mysql> SELECT user,authentication_string,plugin,host FROM mysql.user;

# enable password login
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY 'password';
mysql> FLUSH PRIVILEGES;
mysql> exit;

# should be able to login with password now
mysql -u root -p
 "**********"E "**********"n "**********"t "**********"e "**********"r "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"

mysql>