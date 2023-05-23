#date: 2023-05-23T17:06:25Z
#url: https://api.github.com/gists/34118ad7d3db57da38edffd7bd4e72f3
#owner: https://api.github.com/users/umaparvat

from abc import ABC, abstractmethod
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker

engine = db.create_engine("sqlite:///example.db")
Session = sessionmaker(bind=engine)
session = Session()


class IDataService(ABC):

    @abstractmethod
    def insert(self, data):
        raise NotImplementedError

    @abstractmethod
    def delete(self, condition):
        raise NotImplementedError

    @abstractmethod
    def update(self, data, condition):
        raise NotImplementedError


class Ireader(ABC):

    @abstractmethod
    def read(self, condition):
        raise NotImplementedError


class DatabaseService(Ireader, IDataService):

    def insert(self, data):
        with session as ses:
            ses.insert(data)

    def delete(self, condition):
        with session as ses:
            ses.delete(condition)

    def read(self, condition):
        with session as ses:
            data = ses.query(condition).all()
        return data

    def update(self, data, condition):
        with session as ses:
            ses.update(data, condition)


class DataReader(Ireader):

    def read(self, condition):
        super().read(condition)


# client code
ds = DatabaseService()
ds.insert({"order_id": 1, "order_item": "pen"})
ds.read(condition="order_id=1")

reader = DataReader()
reader.read(condition="order_id=1")
