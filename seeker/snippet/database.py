#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import config

# Асинхронное подключение (для FastAPI)
engine = create_async_engine(config.DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ✅ Синхронное подключение (для миграций и создания таблиц)
sync_engine = create_engine(config.SYNC_DATABASE_URL)
SyncSessionLocal = sessionmaker(sync_engine)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ✅ Экспортируем sync_engine
__all__ = ['engine', 'AsyncSessionLocal', 'sync_engine', 'SyncSessionLocal', 'get_db']