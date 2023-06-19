#date: 2023-06-19T17:06:59Z
#url: https://api.github.com/gists/e7f0c590cdbe0eb63f39a733f950ca52
#owner: https://api.github.com/users/dxphilo

from pydantic import BaseSettings
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv(".env")

class Settings(BaseSettings):
    app_name: str = "octo API"
    NUTRITIONIX_API_ID: str
    NUTRITIONIX_API_KEY: str
    NUTRITIONIX_URL: str
    EXPECTED_CALORIES_PER_DAY: str
    JWT_SECRET: "**********"
    JWT_ALGORITHM:str
    PORT: str
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()