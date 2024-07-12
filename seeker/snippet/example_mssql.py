#date: 2024-07-12T17:00:36Z
#url: https://api.github.com/gists/a444dabf34d6e1da5e05da9cf4a92408
#owner: https://api.github.com/users/DavidRueter

from pymssql import _mssql

class SQLSettings:
    def __init__(self, server= "**********"=1433, user='someuser', password='somepassword',
                 database='somedatabase', appname='someapp', max_conns=10, sql_timeout=120,
                 ):
        self.server = server
        self.port = port
        self.user = user
        self.password = "**********"
        self.database = database
        self.appname = appname
        self.max_conns = max_conns
        self.sql_timeout = sql_timeout

        _mssql.set_max_connections(max_conns)

#set the required SQL Server connection information
sql_settings = "**********"='localhost', database='master', user='sa', password='mypassword')

sql_conn = _mssql.connect(
    server=sql_settings.server,
    port=sql_settings.port,
    user=sql_settings.user,
    password= "**********"
    database=sql_settings.database,
    appname=sql_settings.appname
)

sql_str = 'SELECT TOP 10 * FROM sys.objects'
# Note: you can concatenate your own value for sql_str however you want

sql_conn.execute_query(sql_str)

resultsets = []  # query may return multiple resultsets
first_resultset = []

# get the first resultset and store a list of the rows in this_resultset
this_resultset = [row for row in sql_conn]

# append this_resultset to our list of resultsets (in case the query returns multiple resultsets)
resultsets.append(this_resultset)

# repeat for each additional resultset
have_next_resultset = sql_conn.nextresult()

while have_next_resultset:
    this_resultset = [row for row in sql_conn]
    resultsets.append(this_resultset)
    have_next_resultset = sql_conn.nextresult()

# the resultset is just a list of rows
for row in resultsets[0]:
    # loop through the resultset and do whatever we want
    print(row['object_id'], row['name'])

# note that if there are multiple resultsets you can do
# the same kind of thing with resultsets[1], resultset[2], etc.sultset[2], etc.