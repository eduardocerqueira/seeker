#date: 2022-01-28T16:51:30Z
#url: https://api.github.com/gists/1570a957414a19bb58c35c7f5c14e5c5
#owner: https://api.github.com/users/chand1012

import os
import json

from pymongo import MongoClient
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to the mysql database
mysql_connection = pymysql.connect(host=os.getenv('MYSQL_DB_HOST'),
                                   user=os.getenv('MYSQL_DB_USER'),
                                   password=os.getenv('MYSQL_DB_PASS'),
                                   db=os.getenv('MYSQL_DB_NAME'),
                                   charset='utf8mb4',
                                   cursorclass=pymysql.cursors.DictCursor)

# load tables from os env
tables = os.getenv('MYSQL_DB_TABLES').split(',')

data = {}

for table in tables:
    # get all data from table in mysql
    with mysql_connection.cursor() as cursor:
        sql = "SELECT * FROM {}".format(table)
        cursor.execute(sql)
        data[table] = cursor.fetchall()

# remove UID from all data
for table in data:
    for row in data[table]:
        del row['UID']  

# output all data to json
with open('data.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

# open connection to mongo
client = MongoClient(os.getenv('MONGO_CONNECT_STR'))
db = client[os.getenv('MONGO_DB_NAME')]

# populate each table in mongo
for table in data:
    db[table].insert_many(data[table])

client.close()
mysql_connection.close()