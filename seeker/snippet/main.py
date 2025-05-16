#date: 2025-05-16T17:00:48Z
#url: https://api.github.com/gists/d2534de6568461a01ec24c2d338381d5
#owner: https://api.github.com/users/mypy-play

from cassandra.cluster import Session as CassandraConnection
from typing import cast, Any
from typing_extensions import TypeAlias

TransactionalConnection: TypeAlias = CassandraConnection
x = 42
isinstance(x, CassandraConnection)