#date: 2026-03-16T17:42:17Z
#url: https://api.github.com/gists/cd0f68e1b665833ec1fa02e2742787a0
#owner: https://api.github.com/users/96tm

import os
import warnings
from contextvars import ContextVar
from typing import AsyncGenerator
from unittest import mock

import alembic
import pytest
from alembic.config import Config

# from example.routers.utils.db import get_db
from fastapi import FastAPI
from httpx import AsyncClient  # noqa:
from pytest_factoryboy import register
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from tests.factories import PartCreateSchemaFactory

# Can be overridden by environment variable for testing in CI against other
# database engines

SQLALCHEMY_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "sqlite+aiosqlite:///./tests/files/test.db"
)

# Register factories
register(PartCreateSchemaFactory)

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, echo=True, connect_args={"check_same_thread": False}
)

async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Apply migrations at beginning and end of testing session
@pytest.fixture(scope="function")
def apply_migrations():
    with mock.patch.dict(
        os.environ, {"DATABASE_URL": SQLALCHEMY_DATABASE_URL}, clear=True
    ):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        config = Config("alembic.ini")
        alembic.command.upgrade(config, "head")
        yield
        alembic.command.downgrade(config, "base")


@pytest.fixture()
def app(apply_migrations: None) -> FastAPI:
    """
    Create a fresh database on each test case.
    """
    from app.main import get_application

    return get_application()


@pytest.fixture(scope="function")
@pytest.mark.asyncio
async def db_session(apply_migrations) -> AsyncGenerator[AsyncSession, None]:

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
