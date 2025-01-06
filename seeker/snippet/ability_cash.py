#date: 2025-01-06T16:50:03Z
#url: https://api.github.com/gists/db2e5f9f1a42950bfc29757f48cb1f1f
#owner: https://api.github.com/users/Voronenko

import apsw
# https://pypi.org/project/apsw-sqlite3mc/
# https://github.com/utelle/apsw-sqlite3mc
 
print(apsw.mc_version)
 
con = apsw.Connection("base.cash")
con.pragma("cipher", "aes256cbc")
print(con.pragma("key", "DB_PASSWORD"))

print(f"Database Connection: {con.total_changes}")
 
cursor = con.cursor()
print(f'{cursor=}')
 
# List if tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
 
tables = cursor.fetchall()
print(f'{tables=}')


print("List of tables in the database:")
for table in tables:
    print(table[0])
 
