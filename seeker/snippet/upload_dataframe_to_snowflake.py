#date: 2025-12-25T17:10:41Z
#url: https://api.github.com/gists/08db51f5f07f9ef55a9c885af2e230fd
#owner: https://api.github.com/users/sgouda0412

# Install Packages
### !pip install pandas "snowflake-connector-python[pandas]" sqlalchemy sqlalchemy-snowflake

# Import Packages
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import snowflake.connector
from snowflake.connector.pandas_tools import pd_writer, write_pandas
import datetime

#Create connection to Snowflake using your account and user
credentials = {
"account": '',
"user": '',
"password": "**********"
"database": '',
"schema": '',
"warehouse": '',
"role": ''
}

# Create Connection with snowflake connector
cnx = snowflake.connector.connect(
    account = credentials['account'],
    user = credentials['user'],
    password = "**********"
    database = credentials['database'],
    schema = credentials['schema'],
    warehouse = credentials['warehouse'],
    role = credentials['role']
)

# Create a DataFrame containing data about customers
df = pd.DataFrame([('Mark', 10), ('Luke', 20)], columns=['name', 'balance'])
df["_timestamp"] = datetime.datetime.utcnow()

# Write the data from the DataFrame to the table named "customers".
success, nchunks, nrows, _ = write_pandas(cnx, df, 'customers', auto_create_table=True)

# ----------------------------

engine = create_engine(URL(
    account = credentials['account'],
    user = credentials['user'],
    password = "**********"
    database = credentials['database'],
    schema = credentials['schema'],
    warehouse = credentials['warehouse'],
    role = credentials['role']
))

connection = engine.connect()

try:
    results = connection.execute('select current_version()').fetchone()
    print(results[0])
    df = pd.DataFrame({'test_column': [1, 2, 3], 'test_column_two': [4, 5, 6]})
    df.to_sql(name="test_table", con=connection, if_exists='append', method=pd_writer, index=False)
finally:
    connection.close()
    engine.dispose()


# with engine.connect() as con:

# ProgrammingError: 000904 (42000): SQL compilation error: error line 1 at position 84
# invalid identifier '"test_column"'
"test_column"'
