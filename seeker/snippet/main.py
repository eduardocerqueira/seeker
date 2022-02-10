#date: 2022-02-10T16:52:27Z
#url: https://api.github.com/gists/a12d0c51fb852e679f0fc8b653d4d2eb
#owner: https://api.github.com/users/adriangb

""" Snippet that demonstrates how to use Uvicorn in code.

Feel free to run:

- `python main.py`
"""
import asyncio

import uvicorn
from pydantic import BaseSettings
from xpresso import Path, App


class Config(BaseSettings):
    port: int


async def home(config: Config) -> str:
    return f"Hello World from {config.port}!"


home_path_item = Path("/", get=home)


async def main() -> None:
    config = Config()
    app = App(routes=[home_path_item])
    app.dependency_overrides[Config] = lambda: config
    await uvicorn.Server(uvicorn.Config(app, port=config.port)).serve()


if __name__ == "__main__":
    asyncio.run(main())
