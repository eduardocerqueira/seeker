#date: 2025-07-07T17:03:56Z
#url: https://api.github.com/gists/c4ebff446169d6201e0dbf31217329c7
#owner: https://api.github.com/users/stav888

import os
if os.path.exists('sales.db'):
    os.remove('sales.db')

import sqlite3
conn = sqlite3.connect('sales.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE if not exists sales (
    id INTEGER PRIMARY KEY,
    product TEXT NOT NULL,
    category TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL
);
''')

data = [
    ('Apple', 'Fruit', 10, 1.0),
    ('Banana', 'Fruit', 15, 0.8),
    ('Carrot', 'Vegetable', 12, 0.5),
    ('Lettuce', 'Vegetable', 5, 0.7),
    ('Orange', 'Fruit', 8, 1.2),
    ('Tomato', 'Vegetable', 20, 0.9),
    ('Strawberry', 'Fruit', 6, 2.0)
]

cursor.executemany('''
INSERT INTO sales (product, category, quantity, price)
VALUES (?, ?, ?, ?);
''', data)

cursor.execute('''
SELECT category, SUM(quantity) AS total_quantity
FROM sales
GROUP BY category;
''')

results = cursor.fetchall()
for row in results:
    print(dict(row))

conn.commit()
conn.close()
