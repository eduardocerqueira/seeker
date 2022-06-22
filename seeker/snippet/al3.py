#date: 2022-06-22T17:07:34Z
#url: https://api.github.com/gists/273e91edf1702097e2c343c7c7b80279
#owner: https://api.github.com/users/peytonrunyan

from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Text, Integer
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Dog(Base):
	__tablename__ = 'dogs'

	id = Column(Integer, primary_key=True)
	name = Column(Text, nullable=False, default="Sparky")

if __name__ == "__main__":
	d1, d2 = Dog(), Dog()

	sesh = sessionmaker(bind=create_engine('sqlite:///pets.db'))

	with sesh() as s:
		s.add(d1)
		s.add(d2)
		s.commit()