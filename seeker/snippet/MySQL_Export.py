#date: 2022-03-18T16:47:33Z
#url: https://api.github.com/gists/ed0255128f0e8f284d1789a40b18ba9c
#owner: https://api.github.com/users/rochaandre

import sqlite3 as sql
import os
import csv
from sqlite3 import Error

try:

  # Connect to database
  conn=sql.connect('mydb.db')

  # Create Table into database
  conn.execute('''CREATE TABLE IF NOT EXISTS Employee(Id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,\
               Name TEXT NOT NULL, Salary INT NOT NULL 
              );''')
  # Insert some values to database
  conn.execute('''INSERT INTO Employee(Name, Salary) VALUES('Laxmi', 30000);''')
  conn.execute('''INSERT INTO Employee(Name, Salary) VALUES('Prerna', 40000);''')
  conn.execute('''INSERT INTO Employee(Name, Salary) VALUES('Shweta', 30000);''')
  conn.execute('''INSERT INTO Employee(Name, Salary) VALUES('Soniya', 50000);''')
  conn.execute('''INSERT INTO Employee(Name, Salary) VALUES('Priya', 60000);''')
  conn.commit()

 # To view table data in table format
  print "******Employee Table Data*******"
  cur = conn.cursor()
  cur.execute('''SELECT * FROM Employee''')
  rows = cur.fetchall()
   
  for row in rows:
      print(row)

 # Export data into CSV file
  print "Exporting data into CSV............"
  cursor = conn.cursor()
  cursor.execute("select * from Employee")
  with open("employee_data.csv", "w") as csv_file:
      csv_writer = csv.writer(csv_file, delimiter="\t")
      csv_writer.writerow([i[0] for i in cursor.description])
      csv_writer.writerows(cursor)

  dirpath = os.getcwd() + "/employee_data.csv"
  print "Data exported Successfully into {}".format(dirpath)

except Error as e:
  print(e)

# Close database connection
finally:
  conn.close()