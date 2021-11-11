#date: 2021-11-11T17:13:04Z
#url: https://api.github.com/gists/cea7dfd0996d5ede90b9e0a46ad48e48
#owner: https://api.github.com/users/shirblc

import os
import sys
import os
from sqlalchemy.ext.horizontal_shard import ShardedSession, ShardedQuery
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
)

# Sharding Setup
# -----------------------------------------------------------------
# Based on https://docs.sqlalchemy.org/en/14/_modules/examples/sharding/attribute_shard.html
shards = {
    "read": create_engine(
        os.environ.get("READ_DB_URL")
    ),
    "write": create_engine(
        os.environ.get("WRITE_DB_URL")
    ),
}


def shard_chooser(mapper, instance, clause=None):
    """shard chooser.
    By default returns write since that's the main DB."""
    return "write"


def id_chooser(query, ident):
    """id chooser.

    given a primary key, returns a list of shards
    to search.  here, we don't have any particular information from a
    pk so we just return all shard ids. often, you'd want to do some
    kind of round-robin strategy here so that requests are evenly
    distributed among DBs.
    Adjusted from https://docs.sqlalchemy.org/en/14/_modules/examples/sharding/attribute_shard.html
    """
    if query.lazy_loaded_from:
        # if we are in a lazy load, we can look at the parent object
        # and limit our search to that same shard, assuming that's how we've
        # set things up.
        return [query.lazy_loaded_from.identity_token]
    else:
        return ["read", "write"]


def execute_chooser(query):
    """execute chooser.

    this also returns a list of shard ids, which can
    just be all of them. By default returns the write db
    Adjusted from https://docs.sqlalchemy.org/en/14/_modules/examples/sharding/attribute_shard.html
    """
    return ["write"]


# Engine
Session = sessionmaker(class_=ShardedSession)
Session.configure(
    shards=shards,
    shard_chooser=shard_chooser,
    id_chooser=id_chooser,
    execute_chooser=execute_chooser,
    query_cls=ShardedQuery,
)


# Models
# -----------------------------------------------------------------
BaseModel = declarative_base()


class Category(BaseModel):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    parent_id = Column(Integer, ForeignKey("categories.id"))
