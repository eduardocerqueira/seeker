#date: 2024-06-19T16:43:46Z
#url: https://api.github.com/gists/37cb16a4a076a51f4f5f0d89934e3cae
#owner: https://api.github.com/users/dtrifiro

import asyncio

from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rich.traceback import install


install()

app = FastAPI()


@app.get("/")
def hello_world():
    return JSONResponse(
        dict(
            data="ok",
        )
    )


async def run_http_server(port: int):
    config = Config(
        app=app,
        loop="none",
        port=port,
        host="0.0.0.0",
    )
    server = Server(config)
    await server.serve()


async def run_grpc_server(port: int):
    print("Started grpc_server")
    while True:
        # dummy implementation
        await asyncio.sleep(1)
        print("sleeping")


async def main():
    http = asyncio.create_task(run_http_server(8080))
    grpc = asyncio.create_task(run_grpc_server(8080))

    await http
    await grpc


if __name__ == "__main__":
    asyncio.run(main())
