#date: 2023-07-06T17:04:38Z
#url: https://api.github.com/gists/8da5d839b9d5a691fcd166488f152658
#owner: https://api.github.com/users/dario61081

class Database:
    def __init__(self, **kwargs):
        self.username = kwargs.get('username')
        self.password = "**********"
        self.database = kwargs.get('database')
        self.host = kwargs.get('host')
        self.port = kwargs.get('port')
        self.driver = kwargs.get('driver')
        self.uri = f"{self.driver}: "**********":{self.password}@{self.host}:{self.port}/{self.database}?charset=utf-8"

        self.engine = create_engine(self.uri)
        self.connection = self.engine.connect()
        self.metadata = MetaData(bind=self.engine)
        self.session = Session(bind=self.connection)

    def fetch_all(self, query, *args, **kwargs):
        return self.connection.execute(text(query), *args, **kwargs)rgs, **kwargs)