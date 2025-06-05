#date: 2025-06-05T16:48:57Z
#url: https://api.github.com/gists/b5f5d3ac475a473af56037458ac78c08
#owner: https://api.github.com/users/phoenix7815

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import random
import string

concurrency_limit = 5
timeout = 5

semaphore = asyncio.Semaphore(concurrency_limit)

urls = ['https://books.toscrape.com/'] * 14

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits 
    return ''.join(random.choices(characters, k=length))

async def fetch_data(session, url):
    req_id = generate_random_string(4)
    async with semaphore:
        try:
            async with asyncio.timeout(timeout):
                print(f'Request_Id : {req_id} trying to connect...')
                await asyncio.sleep(2)
                async with session.get(url) as response:
                    if response.status != 200:
                        return url, None, f"HTTP {response.status}"
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    title = soup.title.string if soup.title else "No Title"
                    print(f'Request_Id : {req_id} response get successfully')
                    return url, title, "OK"
        except asyncio.TimeoutError:
            print(f'Request_Id : {req_id} TimeError')
            return url, None, "TimeoutError"
        except Exception as e:
            print(f'Request_Id : {e} Error')
            return url, None, str(e)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f'{i} -> {result}')

if __name__ == "__main__":
    asyncio.run(main())