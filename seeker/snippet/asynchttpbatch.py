#date: 2025-02-18T16:51:51Z
#url: https://api.github.com/gists/77720da347acf022cc25dc5ed3da12a5
#owner: https://api.github.com/users/LX5321

import asyncio
import aiohttp

# Exponential backoff and retry function
async def fetch_with_retries(session, url, retries=5, backoff_factor=1):
    attempt = 0
    while attempt <= retries:
        try:
            async with session.get(url) as response:
                print("Attempt", attempt, "for", url)
                response.raise_for_status()  # Raise exception for HTTP errors
                return await response.json()  # or response.text(), based on API
        except aiohttp.ClientError:
            attempt += 1
            wait_time = backoff_factor * (2 ** (attempt - 1))  # Exponential backoff
            await asyncio.sleep(wait_time)
    return None

# Function to execute API calls in batches
async def execute_api_calls(urls):
    async with aiohttp.ClientSession() as session:
        results = []
        for i in range(0, len(urls), 3):
            group = urls[i:i+3]  # Get the next group of up to 3 URLs
            tasks = [fetch_with_retries(session, url) for url in group]
            group_results = await asyncio.gather(*tasks)  # Execute them in parallel
            results.extend([result for result in group_results if result])  # Collect only successful results
            await asyncio.sleep(1)
        return results

# List of URLs for API calls
urls = [
    'https://jsonplaceholder.typicode.com/posts/1',
    'https://jsonplaceholder.typicode.com/posts/2',
    'https://jsonplaceholder.typicode.com/posts/3',
    'https://jsonplaceholder.typicode.com/posts/4',
    'https://jsonplaceholder.typicode.com/posts/5',
    'https://jsonplaceholder.typicode.com/posts/6',
    'https://jsonplaceholder.typicode.com/posts/7',
    'https://jsonplaceholder.typicode.com/posts/8',
    'https://jsonplaceholder.typicode.com/posts/9',
    'https://jsonplaceholder.typicode.com/posts/999999'
]

# Run the script
data = asyncio.run(execute_api_calls(urls))
print(data)
