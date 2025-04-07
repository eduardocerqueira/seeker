#date: 2025-04-07T17:05:35Z
#url: https://api.github.com/gists/c6425c6e0cc58f824f58716f5cbd0c2e
#owner: https://api.github.com/users/jkeam

from typing import List, Optional
from sqlalchemy import String, ForeignKey, create_engine
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(back_populates="user")
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id = mapped_column(ForeignKey("user_account.id"))
    user: Mapped[User] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"

engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)

with Session(engine) as session:
    # create tables
    Base.metadata.create_all(engine)

    # create user
    user = User()
    user.name = "Jon"
    user.fullname = "Jon Doe"

    # create address
    address = Address()
    address.email_address = "thisisfake@example.com"

    # tie the two objects together
    user.addresses.append(address)

    # save
    #   this save will save both the user and address objects
    #   as well as set the ids correctly
    session.add(user)

    # test and query
    result = session.query(User).filter(User.name == 'Jon').first()
    if result:
        print("--------------------------------------")
        print(result)
        print(f"\tuser id: {result.id}")
        print(result.addresses)
        address = result.addresses[0]
        print(f"\taddress id: {address.id}, user id: {address.user_id}")
        print("--------------------------------------")
    else:
        print("Nothing found")

    session.commit()
