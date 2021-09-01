#date: 2021-09-01T17:14:17Z
#url: https://api.github.com/gists/135cb59092202750b953b3285cc6851d
#owner: https://api.github.com/users/unmade

import asyncio
import itertools

import edgedb
from edgedb.options import IsolationLevel, TransactionOptions, default_backoff


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)



restore_parent_query = """
    WITH
        parents := array_unpack(<array<str>>$parents),
    FOR parent in {parents}
    UNION (
        UPDATE
            File
        FILTER
            .path LIKE parent ++ '/%'
            AND
            .path NOT LIKE parent ++ '/%/%'
        SET {
            parent := (
                SELECT
                    File
                FILTER
                    .path = parent
                LIMIT 1
            )
        }
    )
"""


async def restore_parent(pool, parent):
    async for tx in pool.retrying_transaction():
        async with tx:
            await tx.query(
                restore_parent_query,
                parent=parent,
            )


async with edgedb.create_async_pool(...) as pool:
    pool._options._transaction_options = TransactionOptions(isolation=IsolationLevel.Serializable)
    pool._options._retry_options = RetryOptions(attempts=5, backoff=default_backoff)

    parents = ['path', 'path/path1', 'path/path2', 'path/path1/path', ...] # very large list
    coros = (
        pool.query(
            query,
            parents=[parent for parent in parents if parent],
        )
        for chunk in grouper(parents, 500)
    )
    await asyncio.gather(*coros)