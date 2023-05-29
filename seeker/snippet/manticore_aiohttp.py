#date: 2023-05-29T16:51:31Z
#url: https://api.github.com/gists/55183e3ce2103b203255907d0a083189
#owner: https://api.github.com/users/mrmamongo

import asyncio

import aiohttp


async def insert_doc(index: str, document: dict, host: str = "localhost", port: int = 9308):
    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
        async with session.post(url=f"http://{host}:{port}/insert", json={"index": index, "doc": document}) as resp:
            if resp.status != 200:
                raise RuntimeError("Error while inserting doc: manticore is not available")

            return await resp.json()


async def search(index: str, query: dict, host: str = "localhost", port: int = 9308):
    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
        async with session.post(url=f"http://{host}:{port}/search", json={
            "index": index, "query": query
        }) as resp:
            if resp.status != 200:
                raise RuntimeError("Error while inserting doc: manticore is not available")

            return await resp.json()


async def main():
    await insert_doc(index="j", document={"a": "b"})

    await search(index="j", query={"match": {"a": "b"}})


if __name__ == '__main__':
    asyncio.run(main())
