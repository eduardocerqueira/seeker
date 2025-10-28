#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073



import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # PostgreSQL
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "stock_exchange")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = "**********"
    
    # Security
    SECRET_KEY = "**********"
    ALGORITHM = "HS256"
    
    # Database URLs
    DATABASE_URL = f"postgresql+asyncpg: "**********":{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    SYNC_DATABASE_URL = f"postgresql: "**********":{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

config = Config()SSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

config = Config()