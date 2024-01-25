#date: 2024-01-25T16:45:42Z
#url: https://api.github.com/gists/cbfaa89f4db79bdec859f8ebca5c1b75
#owner: https://api.github.com/users/Alexcooldev

import sqlite3
conn = sqlite3.connect('test3.db')
cursor = conn.cursor()
cursor.execute = ('''CREATE TABLE IF NOT EXISTS mytable (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);''')
insert_data_query = '''INSERT INTO mytable (name, age) VALUES (?, ?);'''
data = [('John', 25), ('Emma', 30), ('Michael', 35)]
cursor.executemany(insert_data_query, data)
conn.commit()
conn.close()
print('Таблица создана и заполнена данными!')