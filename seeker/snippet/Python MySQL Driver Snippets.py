#date: 2022-04-11T16:58:56Z
#url: https://api.github.com/gists/b935a8410257effe67af678232939970
#owner: https://api.github.com/users/poseidon-code

import mysql.connector

db = mysql.connector.connect(
  host="localhost",
  user="sql_username",
  password="sql_user_password",
  database="db_name"     # = "USE <database name>;"
)

cursor = db.cursor()

# show MySQL Users : "SELECT User FROM mysql.user;"
cursor.execute("SELECT User FROM mysql.user")
# (to use a specific user you need to create one first,
# use the MySQL shell to create a user,
'''
sudo mysql -u root -p
CREATE USER 'new_username'@localhost IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON *.* TO 'new_username'@localhost IDENTIFIED BY 'password';
FLUSH PRIVILEGES;
'''
# here, *.* = <db name>.<table name>; grants permissions for specified database and/or table)



# show MySQL Databases : "SHOW DATABASES;"
cursor.execute("SHOW DATABASES")

# create Database : "CREATE DATABASE <database name>;"
cursor.execute("CREATE DATABASE test")
# (to use this database, you need to pre-specify the database name
# inside 'db' object, hence it is better to create a database from MySQL console
'''
mysql -u username -p
CREATE DATABASE <database name>;
'''
# then specify that when initialising 'db' object [3:7])



# show MySQL Tables : "SHOW TABLES;"
cursor.execute("SHOW TABLES")

# show MySQL Tables data : "SELECT * FROM <table name>;"
cursor.execute("SELECT * FROM employees")

# create MySQL Tables
TABLES = {}
TABLES['table_name'] = (
    "CREATE TABLE table_name ("
    "  column_1 int(3) NOT NULL,"
    "  column_2 varchar(14) NOT NULL,"
    "PRIMARY KEY(column_1)"
    ")"
  )
cursor.execute(TABLES["table_name"])

# insert into MySQL Tables
insert_query = (
"INSERT INTO table_name"
"(column_1, column_2)"
"VALUES (%s, %s)"
)
insert_data = (column_1_data, column_2_data)
cursor.execute(insert_query, insert_data)




for x in cursor:
  print(x)


cursor.close()
db.close()