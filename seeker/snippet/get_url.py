#date: 2021-12-03T17:06:46Z
#url: https://api.github.com/gists/f601bb6c7b5bb0eed1663f7b36c5e07e
#owner: https://api.github.com/users/dgtlctzn

async def get_url(session: aiohttp.ClientSession, url: str) -> Dict:
    async with session.get(url) as response:
        return await response.json()