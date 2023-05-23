#date: 2023-05-23T16:59:48Z
#url: https://api.github.com/gists/c26e9774374095860ca9ead82008fe30
#owner: https://api.github.com/users/umaparvat

from abc import ABC, abstractmethod
import  sqlalchemy as db
from sqlalchemy.orm import sessionmaker

engine = db.create_engine("sqlite:///example.db")
Session = sessionmaker(bind=engine)
session = Session()

class DatabaseService:

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


class DataReader(DatabaseService):

    def read(self, condition):
        super().read(condition)

    def insert(self, data):
        raise NotImplementedError

    def update(self, data, condition):
        raise NotImplementedError

    def delete(self, condition):
        raise NotImplementedError

# client code
ds = DatabaseService()
ds.insert({"order_id":1, "order_item": "pen"})
ds.read(condition="order_id=1")

reader = DataReader()
reader.read(condition="order_id=1")
reader.insert({"order_id":2, "order_item": "pencil"})  # this will throw error