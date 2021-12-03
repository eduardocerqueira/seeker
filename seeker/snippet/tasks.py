#date: 2021-12-03T17:03:59Z
#url: https://api.github.com/gists/c80c588a9ae29dcd8914260f9f744ae8
#owner: https://api.github.com/users/dgtlctzn

async def request_urls(urls: List[str]):
    async with aiohttp.ClientSession() as session:
        tasks: List[asyncio.Task] = []
        for url in urls:
            tasks.append(
                asyncio.ensure_future(
                    get_url(session, url)
                )
            )
        return await asyncio.gather(*tasks)