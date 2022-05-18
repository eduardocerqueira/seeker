#date: 2022-05-18T17:11:26Z
#url: https://api.github.com/gists/caa20092133842b127de0fdd31d74024
#owner: https://api.github.com/users/realnhchi

import sys
import os
import json
import asyncio
import aiohttp


# Initialize connection pool
conn = aiohttp.TCPConnector(limit_per_host=100, limit=0, ttl_dns_cache=300)
PARALLEL_REQUESTS = 100
results = []
urls = ['https://jsonplaceholder.typicode.com/todos/1' for i in range(10)] #array of urls

async def gather_with_concurrency(n):
    semaphore = asyncio.Semaphore(n)
    session = aiohttp.ClientSession(connector=conn)

    # heres the logic for the generator
    async def get(url):
        async with semaphore:
            async with session.get(url, ssl=False) as response:
                obj = json.loads(await response.read())
                results.append(obj)
    await asyncio.gather(*(get(url) for url in urls))
    await session.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(gather_with_concurrency(PARALLEL_REQUESTS))
conn.close()

print(f"Completed {len(urls)} requests with {len(results)} results")