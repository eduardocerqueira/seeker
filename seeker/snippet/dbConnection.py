#date: 2021-09-08T17:00:26Z
#url: https://api.github.com/gists/edf1ce3f488261afddef68f9dbdf3043
#owner: https://api.github.com/users/ZenithClown

# -*- encoding: utf-8 -*-

__author__       = "Debmalya Pramanik"

__status__       = "production"
__version__      = "1.0"
__docformat__    = "camelCasing"

__copyright__    = "Copyright (c) 2020 Debmalya Pramanik"
__affiliation__  = "Indian Institute of Technology (IIT), Dhanbad"


import pymysql
import pymssql
import sqlalchemy as sa

class DBConnection(object):
    """Class to Connect to all the Popular Database using Python

    This is a general purpose connection object, that can be used to connect to any type of
    database using `python` SQL connection libraries. Currently, the class is defined for MySQL and MSSQL Server,
    but given the simplicity of the code, all different types of database can be easily controlled.

    :param username: Username for connecting to the database.
    :param password: Authenticationpassword.
    :param hostname: Hostname IP address, defaults to `locahost`
    :param database: Name of the database.

    Keyword Arguments
    :param port: Port at which the database is accessible. By default it connects to 3306 port for MySQL server,
                 and 1433 port for MSSQL server.
    :param db_name: Name of the database to connect to. Defaults to MySQL Server.

    Example(s): The following can be used to understand the connect and run query for any type of database:

    ```python
      obj = DBConnection(username = "root", password = "admin", database = "dummy", port = 10002)
      con = obj.connect()

      result = con.execute("SELECT * FROM `table-name`")
      for row in result:
        print(row)

      con.close()
    ```

    Configure other keyword arguments to control any other parameters.
    """

    def __init__( self, username : str, password : str, hostname : str = 'localhost', database : str = 'database', **kwargs):
        # default constructor

        self.username = username
        self.password = password
        self.hostname = hostname
        self.database = database

        # keyword arguments
        self._db   = self.db_name(kwargs.get("db_name", "mysql"))
        self._port = kwargs.get("port", self._default_ports(self._db))
        self._conf = self._default_connectors(self._db) # default sa engine


    def db_name(self, name : str) -> str:
        """Format DB-Name and Return - for Consistency"""

        return name.lower()


    def _default_ports(self, dbName : str) -> int:
        # defines default port for populat db engines

        return {
            "mysql" : 3306,
            "mssql" : 1433
        }.get(dbName)


    def _default_connectors(self, name : str) -> str:
        # defines the default connectors

        return {
            "mysql" : "mysql+pymysql",
            "mssql" : "mssql+pymssql"
        }.get(name)


    @property
    def connectionString(self) -> str:
        """General Connection String for any DB Connection"""

        return f"{self.username}:{self.password}@{self.hostname}:{self._port}/{self.database}"


    @property
    def engineDialect(self):
        # {connectors} + {connection string}
        # https://docs.sqlalchemy.org/en/14/core/engines.html
        
        return f"{self._conf}://{self.connectionString}"


    @property
    def engine(self):
        return sa.create_engine(self.engineDialect)


    def connect(self):
        connection = self.engine.connect()
        return connection
    

    def __repr__(self):
        # representation

        return f"Connecting to {self.connectionString}"
