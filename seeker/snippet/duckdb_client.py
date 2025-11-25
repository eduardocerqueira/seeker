#date: 2025-11-25T16:53:58Z
#url: https://api.github.com/gists/f6103d0e6ad067dee411e43550294af7
#owner: https://api.github.com/users/Insighttful

#!/usr/bin/env python3.13

"""
MIT License
For the full text of the MIT License, please visit:
https://opensource.org/licenses/MIT
"""

"""Async DuckDB client with lightweight pooling and helpers leveraging aioduckdb."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypeVar
from pathlib import Path
from contextlib import asynccontextmanager
from collections.abc import Callable, Sequence, Awaitable, AsyncIterator

import aioduckdb
from aioduckdb import Connection
from fastapi import Request

from shared.environment import Environment, get_environment


LOGGER = logging.getLogger("duckdb.client")
APP_STATE_KEY = "duckdb_client"

T = TypeVar("T")


class AsyncDuckDBClient:
    """Manage pooled asynchronous connections to DuckDB."""

    def __init__(
        self,
        env: Environment | None = None,
        *,
        database: str | Path | None = None,
    ) -> None:
        self.env = env or get_environment()
        self._database = str(database or self.env.duckdb_path)
        self._pool_size = self.env.duckdb_pool_size
        self._pool: asyncio.LifoQueue[Connection] | None = None
        self._startup_lock = asyncio.Lock()
        self._checked_out: set[Connection] = set()
        self._shutting_down = False

    async def startup(self) -> None:
        """Create the connection pool if it has not been initialized."""

        if self._pool is not None:
            return

        async with self._startup_lock:
            if self._pool is not None:
                return

            if self._database not in {":memory:", "memory"}:
                path = Path(self._database)
                path.parent.mkdir(parents=True, exist_ok=True)

            self._pool = asyncio.LifoQueue(maxsize=self._pool_size)
            for _ in range(self._pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)

    async def _checkpoint_database(self) -> None:
        if self._database in {":memory:", "memory"}:
            return

        try:
            conn = await aioduckdb.connect(database=self._database, read_only=False)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed to open DuckDB connection for checkpoint", exc_info=exc
            )
            return

        try:
            await conn.execute("CHECKPOINT")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to checkpoint DuckDB database", exc_info=exc)
        finally:
            await conn.close()

        LOGGER.info("DuckDB checkpoint completed (database=%s)", self._database)

    async def shutdown(self) -> None:
        """Dispose of all pooled connections."""

        self._shutting_down = True
        pool = self._pool
        self._pool = None

        to_close: list[Connection] = []
        if pool is not None:
            while not pool.empty():
                to_close.append(pool.get_nowait())

        to_close.extend(self._checked_out)
        self._checked_out.clear()

        for conn in to_close:
            try:
                await conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to close DuckDB connection", exc_info=exc)

        await self._checkpoint_database()

        self._shutting_down = False
        LOGGER.info("DuckDB pool shut down")

    async def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Execute a statement and eagerly fetch all rows."""

        async with self.connection() as conn:
            cursor = await conn.execute(sql, parameters or ())
            try:
                rows = list(await cursor.fetchall())
            finally:
                await cursor.close()
            return rows

    async def executemany(
        self,
        sql: str,
        seq_of_parameters: Sequence[Sequence[Any]],
    ) -> None:
        """Execute a statement for many parameter sets."""

        async with self.connection() as conn:
            cursor = await conn.executemany(sql, seq_of_parameters)
            await cursor.close()

    async def run_transaction(
        self,
        func: Callable[[Connection], Awaitable[T]],
    ) -> T:
        """Run a coroutine within an explicit transaction boundary."""

        async with self.connection() as conn:
            await conn.execute("BEGIN TRANSACTION")
            try:
                result = await func(conn)
            except Exception:
                await conn.execute("ROLLBACK")
                raise
            else:
                await conn.execute("COMMIT")
                return result

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Connection]:
        """Yield a live DuckDB connection from the pool."""

        conn = await self._acquire_connection()
        try:
            yield conn
        finally:
            await self._release_connection(conn)

    async def _create_connection(self) -> Connection:
        conn = await aioduckdb.connect(database=self._database, read_only=False)
        await conn.execute(
            "SET memory_limit = ?",
            (f"{self.env.duckdb_memory_limit_mb}MB",),
        )
        await conn.execute("SET threads = ?", (self.env.duckdb_threads,))

        if self.env.duckdb_load_spatial:
            await conn.execute("INSTALL spatial")
            await conn.execute("LOAD spatial")

        return conn

    async def _acquire_connection(self) -> Connection:
        if self._pool is None:
            raise RuntimeError("DuckDB client not started")

        conn = await self._pool.get()
        self._checked_out.add(conn)
        return conn

    async def _release_connection(self, conn: Connection) -> None:
        self._checked_out.discard(conn)
        if self._shutting_down:
            await conn.close()
            return

        if self._pool is None:
            await conn.close()
            return

        await self._pool.put(conn)


def get_duckdb_client(request: Request) -> AsyncDuckDBClient:
    """FastAPI dependency that returns the DuckDB client from app.state."""

    client = getattr(request.app.state, APP_STATE_KEY, None)
    if not isinstance(client, AsyncDuckDBClient):
        raise RuntimeError("DuckDB client not initialized")
    return client
