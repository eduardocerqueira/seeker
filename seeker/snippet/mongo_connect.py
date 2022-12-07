#date: 2022-12-07T16:45:25Z
#url: https://api.github.com/gists/0abd3433f716657a4902e20670f754b7
#owner: https://api.github.com/users/iKunalChhabra

import pymongo
import pandas as pd


class Mongo:
    def __init__(self, db_name, host, port):
        self.__db_name = db_name
        self.__host = host
        self.__port = port
        self.__collection = None

    @property
    def collection(self):
        if self.__collection is None:
            raise Exception("Collection not initialized")
        return self.__collection

    @collection.setter
    def collection(self, collection_name):
        client = pymongo.MongoClient(self.__host, self.__port)
        db = client[self.__db_name]
        self.__collection = db[collection_name]

    def upsert(self, pk, data):
        print('\nUpserting data')
        print('PK: ', pk)
        print('Data: ', data)
        self.collection.update_one(pk, {'$set': data}, upsert=True)
        print('Upserted data')

    def find(self, filter={}, type='dict'):
        print('\nFinding data')
        print('filter: ', filter)
        data_dict = self.collection.find(filter)
        print('Found data')
        if type == 'dict':
            return data_dict
        elif type == 'DataFrame':
            return pd.DataFrame(list(data_dict)).drop('_id', axis=1)

    def count(self, filter={}):
        print('\nCounting data')
        print('filter: ', filter)
        count =  self.collection.count_documents(filter)
        print('Counted data')
        return count

    def delete(self, filter):
        print('\nDeleting data')
        print('filter: ', filter)
        if not filter and type(filter) == dict:
            raise Exception("Cannot delete all documents")

        self.collection.delete_many(filter)
        print('Deleted data')


if __name__ == '__main__':
    mongo = Mongo(db_name='test', host='localhost', port=27017)
    mongo.collection = 'myapp'

    count = mongo.count(filter={'country': 'India'})
    print('Count: ', count)

    mongo.delete(filter={'country': 'India'})

    mongo.upsert(pk={'id': 44}, data={'name': 'Sam', 'age': 25, 'country': 'Australia'})
    mongo.upsert(pk={'id': 66}, data={'name': 'Rahul', 'age': 44, 'country': 'India'})
    mongo.upsert(pk={'id': 77}, data={'name': 'Jim', 'age': 56, 'country': 'USA'})

    df = mongo.find(type='DataFrame')
    print(df)
