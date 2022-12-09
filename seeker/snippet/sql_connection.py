#date: 2022-12-09T17:04:58Z
#url: https://api.github.com/gists/257f9ebd60c4f3153f43b98015e9cdcd
#owner: https://api.github.com/users/kaustubh-26

import pypyodbc as odbc # pip install pypyodbc

# Database Credentials
DRIVER_NAME = 'SQL SERVER'
SERVER_NAME = '<server_name>\SQLEXPRESS'
DATABASE_NAME = 'database_name'

# Database connection string
connection_string = f"""
    DRIVER={{{DRIVER_NAME}}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Trust_Connection=yes;
"""
# Create connection object
conn = odbc.connect(connection_string)
print(conn)

cursor = conn.cursor()

# Result set
rs = cursor.execute("SELECT * FROM dbo.table_name")

for row in rs:
    print(row[0])
