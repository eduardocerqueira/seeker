#date: 2025-09-16T16:53:09Z
#url: https://api.github.com/gists/1db69a572dbc4ac9b7183c130e721327
#owner: https://api.github.com/users/nt7nguye

from enum import Enum
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from pydantic import BaseModel


class Customer(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    accounts: List["Account"] = Relationship(back_populates="customer")


class Account(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    customer_id: str = Field(foreign_key="customer.id")
    customer: Customer = Relationship(back_populates="accounts")
    tb_account_id: int = Field(unique=True, index=True)
